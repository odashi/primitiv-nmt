#include "config.h"

#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <primitiv/primitiv.h>
#ifdef MYMT_USE_CUDA
#include <primitiv/primitiv_cuda.h>
#endif

#include "lstm.h"
#include "mymt.pb.h"
#include "sampler.h"
#include "utils.h"
#include "vocabulary.h"

using namespace std;

class EncoderDecoder {
  string name_;
  unsigned src_vocab_size_, trg_vocab_size_;
  unsigned embed_size_, hidden_size_;
  float dropout_rate_;
  primitiv::Parameter enc_plxe_, dec_plxe_, dec_pwhy_, dec_pby_;
  ::LSTM enc_rnn_, dec_rnn_;
  primitiv::Node dec_lxe_, dec_why_, dec_by_;

public:
  EncoderDecoder(const string &name,
      unsigned src_vocab_size, unsigned trg_vocab_size,
      unsigned embed_size, unsigned hidden_size,
      float dropout_rate)
    : name_(name)
    , src_vocab_size_(src_vocab_size)
    , trg_vocab_size_(trg_vocab_size)
    , embed_size_(embed_size)
    , hidden_size_(hidden_size)
    , dropout_rate_(dropout_rate)
    , enc_plxe_(name_ + ".enc.lxe", {embed_size_, src_vocab_size_},
        primitiv::initializers::XavierUniform())
    , dec_plxe_(name_ + ".dec.lxe", {embed_size_, trg_vocab_size_},
        primitiv::initializers::XavierUniform())
    , dec_pwhy_(name_ + ".dec.why", {trg_vocab_size_, hidden_size},
        primitiv::initializers::XavierUniform())
    , dec_pby_(name_ + ".dec.by", {trg_vocab_size_},
        primitiv::initializers::Constant(0))
    , enc_rnn_(name_ + ".enc.rnn", embed_size_, hidden_size_)
    , dec_rnn_(name_ + ".dec.rnn", embed_size_, hidden_size_) {}

  // Loads all parameters.
  EncoderDecoder(const string &name, const string &prefix)
    : name_(name)
    , enc_plxe_(primitiv::Parameter::load(prefix + name_ + ".enc.lxe"))
    , dec_plxe_(primitiv::Parameter::load(prefix + name_ + ".dec.lxe"))
    , dec_pwhy_(primitiv::Parameter::load(prefix + name_ + ".dec.why"))
    , dec_pby_(primitiv::Parameter::load(prefix + name_ + ".dec.by"))
    , enc_rnn_(name_ + ".enc.rnn", prefix)
    , dec_rnn_(name_ + ".dec.rnn", prefix) {
      ifstream ifs;
      ::open_file(prefix + name_ + ".config", ifs);
      ifs >> src_vocab_size_;
      ifs >> trg_vocab_size_;
      ifs >> embed_size_;
      ifs >> hidden_size_;
      ifs >> dropout_rate_;
    }

  // Saves all parameters.
  void save(const string &prefix) const {
    enc_plxe_.save(prefix + enc_plxe_.name());
    dec_plxe_.save(prefix + dec_plxe_.name());
    dec_pwhy_.save(prefix + dec_pwhy_.name());
    dec_pby_.save(prefix + dec_pby_.name());
    enc_rnn_.save(prefix);
    dec_rnn_.save(prefix);
    ofstream ofs;
    ::open_file(prefix + name_ + ".config", ofs);
    ofs << src_vocab_size_ << endl;
    ofs << trg_vocab_size_ << endl;
    ofs << embed_size_ << endl;
    ofs << hidden_size_ << endl;
    ofs << dropout_rate_ << endl;
  }

  // Adds parameters to the trainer.
  void register_training(primitiv::Trainer &trainer) {
    trainer.add_parameter(enc_plxe_);
    trainer.add_parameter(dec_plxe_);
    trainer.add_parameter(dec_pwhy_);
    trainer.add_parameter(dec_pby_);
    enc_rnn_.register_training(trainer);
    dec_rnn_.register_training(trainer);
  }

  // Encodes source batch and initializes decoder states.
  void encode(
      const vector<vector<unsigned>> &src_batch, bool train) {
    namespace F = primitiv::node_ops;
    using primitiv::Node;

    // Reversed encoding.
    Node enc_lxe = F::input(enc_plxe_);
    enc_rnn_.init();
    for (auto it = src_batch.rbegin(); it != src_batch.rend(); ++it) {
      Node e = F::pick(enc_lxe, *it, 1);
      e = F::dropout(e, dropout_rate_, train);
      enc_rnn_.forward(e);
    }

    // Initializes decoder states.
    dec_lxe_ = F::input(dec_plxe_);
    dec_why_ = F::input(dec_pwhy_);
    dec_by_ = F::input(dec_pby_);
    dec_rnn_.init(enc_rnn_.get_c(), enc_rnn_.get_h());
  }

  // One-step decoding
  primitiv::Node decode_step(
      const vector<unsigned> &trg_words, bool train) {
    namespace F = primitiv::node_ops;
    using primitiv::Node;

    Node e = F::pick(dec_lxe_, trg_words, 1);
    e = F::dropout(e, dropout_rate_, train);
    Node h = dec_rnn_.forward(e);
    h = F::dropout(h, dropout_rate_, train);
    return F::matmul(dec_why_, h) + dec_by_;
  }

  // Calculates the loss function.
  primitiv::Node loss(
      const vector<vector<unsigned>> &trg_batch, bool train) {
    namespace F = primitiv::node_ops;
    using primitiv::Node;

    vector<Node> losses;
    for (unsigned i = 0; i < trg_batch.size() - 1; ++i) {
      Node y = decode_step(trg_batch[i], train);
      losses.emplace_back(F::softmax_cross_entropy(y, trg_batch[i + 1], 0));
    }
    return F::batch::mean(F::sum(losses));
  }
};

float process(
    ::EncoderDecoder &model, primitiv::Trainer &trainer,
    const mymt::proto::Corpus &corpus, unsigned batch_size, bool train) {
  random_device rd;
  ::RandomBatchSampler sampler(corpus, batch_size, rd());
  unsigned num_sents = 0;
  unsigned num_labels = 0;
  float accum_loss = 0;

  while (sampler.has_next()) {
    Batch batch = sampler.next();
    const unsigned actual_batch_size = batch.source[0].size();

    primitiv::Graph g;
    primitiv::Graph::set_default_graph(g);
    model.encode(batch.source, train);
    const primitiv::Node loss = model.loss(batch.target, train);
    accum_loss += g.forward(loss).to_vector()[0] * actual_batch_size;

    if (train) {
      trainer.reset_gradients();
      g.backward(loss);
      trainer.update();
    }

    num_sents += actual_batch_size;
    num_labels += actual_batch_size * (batch.target.size() - 1);
    cout << num_sents << '/' << corpus.samples_size() << '\r' << flush;
  }

  return accum_loss / num_labels;
}

vector<string> infer(
    ::EncoderDecoder &model, const mymt::proto::Corpus &corpus,
    ::Vocabulary trg_vocab) {
  const unsigned bos_id = trg_vocab.stoi("<bos>");
  const unsigned eos_id = trg_vocab.stoi("<eos>");
  vector<string> hyps;

  for (const auto &sample : corpus.samples()) {
    // Make source batch
    const auto &src_ids = sample.source().token_ids();
    vector<vector<unsigned>> src_batch(src_ids.size(), vector<unsigned>(1));
    for (int i = 0; i < src_ids.size(); ++i) src_batch[i][0] = src_ids[i];

    // Initialize the model
    primitiv::Graph g;
    primitiv::Graph::set_default_graph(g);
    model.encode(src_batch, false);
    vector<unsigned> trg_ids {bos_id};

    // Decode
    while (trg_ids.back() != eos_id) {
      const vector<unsigned> prev {trg_ids.back()};
      const primitiv::Node scores = model.decode_step(prev, false);
      const unsigned next = ::argmax(g.forward(scores).to_vector());
      trg_ids.emplace_back(next);

      if (trg_ids.size() == 64 + 1) {
        trg_ids.emplace_back(eos_id);
        break;
      }
    }

    // Make resulting string.
    string hyp;
    for (unsigned i = 1; i < trg_ids.size() - 1; ++i) {
      if (i > 1) hyp += ' ';
      hyp += trg_vocab.itos(trg_ids[i]);
    }
    hyps.emplace_back(move(hyp));
  }

  return hyps;
}

void save_score(const string &path, float score) {
  ofstream ofs;
  ::open_file(path, ofs);
  char buf[16];
  ::sprintf(buf, "%.8e", score);
  ofs << buf << endl;
}

void save_hyps(const string &path, const vector<string> &hyps) {
  ofstream ofs;
  ::open_file(path, ofs);
  for (const string &hyp : hyps) ofs << hyp << endl;
}

int main(int argc, char *argv[]) {
  ::check_args(argc, argv, {
      "(int) Embedding size",
      "(int) Hidden size",
      "(float) dropout_rate",
      "(int) Batch size",
      "(file/in) Train corpus file",
      "(file/in) Dev corpus file",
      "(file/in) Source vocabulary file",
      "(file/in) Target vocabulary file",
      "(dir/out) Model directory",
      "(int) Number of epochs"
  });

  ::global_try_block([&]() {
      const unsigned embed_size = stoi(*++argv);
      const unsigned hidden_size = stoi(*++argv);
      const float dropout_rate = stof(*++argv);
      const unsigned batch_size = stoi(*++argv);
      const string train_corpus_file = *++argv;
      const string dev_corpus_file = *++argv;
      const string src_vocab_file = *++argv;
      const string trg_vocab_file = *++argv;
      const string model_dir = *++argv;
      const unsigned num_epochs = stoi(*++argv);

      cout << "Loading vocabularies ... " << flush;
      const ::Vocabulary src_vocab(src_vocab_file);
      const ::Vocabulary trg_vocab(trg_vocab_file);
      cout << "done." << endl;

      cout << "Loading corpus ... " << flush;
      mymt::proto::Corpus train_corpus, dev_corpus;
      ::load_proto(train_corpus_file, train_corpus);
      ::load_proto(dev_corpus_file, dev_corpus);
      cout << "done." << endl;

      cout << "Initializing devices ... " << flush;
#ifdef MYMT_USE_CUDA
      primitiv::CUDADevice dev(0);
#else
      primitiv::CPUDevice dev;
#endif
      primitiv::Device::set_default_device(dev);
      cout << "done." << endl;

      cout << "Initializing model ... " << flush;
      ::EncoderDecoder model(
          "encdec", src_vocab.size(), trg_vocab.size(),
          embed_size, hidden_size, dropout_rate);
      cout << "done." << endl;

      cout << "Initializing trainer ... " << flush;
      primitiv::trainers::Adam trainer;
      trainer.set_weight_decay(1e-6);
      trainer.set_gradient_clipping(5);
      model.register_training(trainer);
      cout << "done." << endl;

      cout << "Saving initial model ... " << flush;
      {
        const string subdir = model_dir + "/" + ::get_epoch_str(0);
        ::make_directory(subdir);
        model.save(subdir + "/model.");
        trainer.save(subdir + "/trainer");
      }
      cout << "done." << endl;

      cout << "Start training." << endl;
      for (unsigned epoch = 1; epoch <= num_epochs; ++epoch) {
        cout << "Epoch " << epoch << ':' << endl;
        float train_avg_loss = ::process(model, trainer, train_corpus, batch_size, true);
        cout << "  Train loss: " << train_avg_loss << endl;
        float dev_avg_loss = ::process(model, trainer, dev_corpus, 1, false);
        cout << "  Dev loss: " << dev_avg_loss << endl;
        cout << "  Generating dev hyps ... " << flush;
        const vector<string> dev_hyps = ::infer(model, dev_corpus, trg_vocab);
        cout << "done." << endl;

        cout << "  Saving current model ... " << flush;
        const string subdir = model_dir + "/" + ::get_epoch_str(epoch);
        ::make_directory(subdir);
        model.save(subdir + "/model.");
        trainer.save(subdir + "/trainer");
        ::save_score(subdir + "/train.avg_loss", train_avg_loss);
        ::save_score(subdir + "/dev.avg_loss", dev_avg_loss);
        ::save_hyps(subdir + "/dev.hyp", dev_hyps);
        cout << "done." << endl;
      }
      cout << "Finished." << endl;
  });

  return 0;
}
