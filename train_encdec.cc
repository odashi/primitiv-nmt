#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <primitiv/primitiv.h>
#include <primitiv/primitiv_cuda.h>

#include "lstm.h"
#include "mymt.pb.h"
#include "sampler.h"
#include "utils.h"
#include "vocabulary.h"

using namespace std;

class EncoderDecoder {
  std::string name_;
  unsigned src_vocab_size_, trg_vocab_size_;
  unsigned embed_size_, hidden_size_;
  float dropout_rate_;
  primitiv::Parameter enc_plxe_, dec_plxe_, dec_pwhy_, dec_pby_;
  ::LSTM enc_rnn_, dec_rnn_;
  primitiv::Node dec_lxe_, dec_why_, dec_by_;

public:
  EncoderDecoder(const std::string &name,
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
  EncoderDecoder(const std::string &name, const std::string &prefix)
    : name_(name)
    , enc_plxe_(primitiv::Parameter::load(prefix + name_ + ".enc.lxe"))
    , dec_plxe_(primitiv::Parameter::load(prefix + name_ + ".dec.lxe"))
    , dec_pwhy_(primitiv::Parameter::load(prefix + name_ + ".dec.why"))
    , dec_pby_(primitiv::Parameter::load(prefix + name_ + ".dec.by"))
    , enc_rnn_(name_ + ".enc.rnn", prefix)
    , dec_rnn_(name_ + ".dec.rnn", prefix) {
      std::ifstream ifs;
      ::open_file(prefix + name_ + ".config", ifs);
      ifs >> src_vocab_size_;
      ifs >> trg_vocab_size_;
      ifs >> embed_size_;
      ifs >> hidden_size_;
      ifs >> dropout_rate_;
    }

  // Saves all parameters.
  void save(const std::string &prefix) const {
    enc_plxe_.save(prefix + enc_plxe_.name());
    dec_plxe_.save(prefix + dec_plxe_.name());
    dec_pwhy_.save(prefix + dec_pwhy_.name());
    dec_pby_.save(prefix + dec_pby_.name());
    enc_rnn_.save(prefix);
    dec_rnn_.save(prefix);
    std::ofstream ofs;
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
      const std::vector<std::vector<unsigned>> &src_batch, bool train) {
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
      const std::vector<unsigned> &trg_words, bool train) {
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
      const std::vector<std::vector<unsigned>> &trg_batch, bool train) {
    namespace F = primitiv::node_ops;
    using primitiv::Node;

    std::vector<Node> losses;
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
  std::random_device rd;
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

void save_score(const std::string &path, float score) {
  std::ofstream ofs;
  ::open_file(path, ofs);
  char buf[16];
  ::sprintf(buf, "%.8e", score);
  ofs << buf << endl;
}

void save_model(
    unsigned epoch, const std::string &model_dir,
    const ::EncoderDecoder &model,
    const primitiv::Trainer &trainer,
    float train_avg_loss, float dev_avg_loss) {
  const std::string subdir = model_dir + "/" + ::get_epoch_str(epoch);
  ::make_directory(subdir);
  model.save(subdir + "/model.");
  trainer.save(subdir + "/trainer");
  ::save_score(subdir + "/train.avg_loss", train_avg_loss);
  ::save_score(subdir + "/dev.avg_loss", dev_avg_loss);
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
      primitiv::CUDADevice dev(0);
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
      ::save_model(0, model_dir, model, trainer, 1e10, 1e10);
      cout << "done." << endl;

      cout << "Start training." << endl;
      for (unsigned epoch = 1; epoch <= num_epochs; ++epoch) {
        cout << "Epoch " << epoch << ':' << endl;
        float train_avg_loss = ::process(
            model, trainer, train_corpus, batch_size, true);
        cout << "  Train loss: " << train_avg_loss << endl;
        float dev_avg_loss = ::process(
            model, trainer, dev_corpus, 1, false);
        cout << "  Dev loss: " << dev_avg_loss << endl;
        cout << "  Saving current model ... " << flush;
        ::save_model(
            epoch, model_dir, model, trainer, train_avg_loss, dev_avg_loss);
        cout << "done." << endl;
      }
      cout << "Finished." << endl;
  });

  return 0;
}
