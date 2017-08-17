#ifndef MYMT_NMT_UTILS_H_
#define MYMT_NMT_UTILS_H_

#include <iostream>
#include <string>
#include <vector>

#include <primitiv/primitiv.h>

#include "attention_encoder_decoder.h"
#include "sampler.h"
#include "utils.h"
#include "vocabulary.h"

inline float process(
    ::AttentionEncoderDecoder &model, primitiv::Trainer &trainer,
    ::Sampler &sampler, bool train) {
  unsigned num_sents = 0;
  unsigned num_labels = 0;
  float accum_loss = 0;
  sampler.reset();

  while (sampler.has_next()) {
    const Batch batch = sampler.next();
    const unsigned batch_size = batch.source[0].size();

    primitiv::Graph g;
    primitiv::Graph::set_default_graph(g);
    model.encode(batch.source, train);
    const primitiv::Node loss = model.loss(batch.target, train);
    accum_loss += g.forward(loss).to_vector()[0] * batch_size;

    if (train) {
      trainer.reset_gradients();
      g.backward(loss);
      trainer.update();
    }

    num_sents += batch_size;
    num_labels += batch_size * (batch.target.size() - 1);
    std::cout << num_sents << '/' << sampler.num_sentences()
              << '\r' << std::flush;
  }

  return accum_loss / num_labels;
}

inline std::string infer_sentence(
    ::AttentionEncoderDecoder &model, const ::Vocabulary &trg_vocab,
    const std::vector<std::vector<unsigned>> &src_batch,
    unsigned limit) {
  namespace F = primitiv::node_ops;
  using primitiv::Node;

  const unsigned bos_id = trg_vocab.stoi("<bos>");
  const unsigned eos_id = trg_vocab.stoi("<eos>");

  // Initialize the model
  primitiv::Graph g;
  primitiv::Graph::set_default_graph(g);
  model.encode(src_batch, false);
  std::vector<unsigned> trg_ids {bos_id};

  // Decode
  while (trg_ids.back() != eos_id) {
    const std::vector<unsigned> prev {trg_ids.back()};
    const Node a_logits = model.decode_atten(prev, false);
    const Node a_probs = F::softmax(a_logits, 0);
    const Node scores = model.decode_word(a_probs);
    const unsigned next = ::argmax(g.forward(scores).to_vector());
    trg_ids.emplace_back(next);

    if (trg_ids.size() == limit + 1) {
      trg_ids.emplace_back(eos_id);
      break;
    }
  }

  // Make resulting string.
  std::string hyp;
  for (unsigned i = 1; i < trg_ids.size() - 1; ++i) {
    if (i > 1) hyp += ' ';
    hyp += trg_vocab.itos(trg_ids[i]);
  }

  return hyp;
}

inline std::vector<std::string> infer_corpus(
    ::AttentionEncoderDecoder &model, const ::Vocabulary &trg_vocab,
    ::Sampler &sampler) {
  std::vector<std::string> hyps;
  sampler.reset();

  while (sampler.has_next()) {
    const Batch batch = sampler.next();
    const unsigned batch_size = batch.source[0].size();
    if (batch_size != 1) {
      throw std::runtime_error(
          "inference is allowed only for each one sentence.");
    }
    hyps.emplace_back(::infer_sentence(model, trg_vocab, batch.source, 64));
  }

  return hyps;
}

inline void save_all(
    const std::string &model_dir,
    const ::AttentionEncoderDecoder &model,
    const primitiv::Trainer &trainer,
    float train_avg_loss,
    float dev_avg_loss,
    const std::vector<std::string> &dev_hyps) {
  ::make_directory(model_dir);
  model.save(model_dir + "/model.");
  trainer.save(model_dir + "/trainer");
  ::save_value(model_dir + "/train.avg_loss", train_avg_loss);
  ::save_value(model_dir + "/dev.avg_loss", dev_avg_loss);
  ::save_strings(model_dir + "/dev.hyp", dev_hyps);
}

inline void train_step(
    unsigned epoch,
    const std::string &model_dir, ::AttentionEncoderDecoder &model,
    const ::Vocabulary &trg_vocab, primitiv::Trainer &trainer,
    ::Sampler &train_sampler, ::Sampler &dev_sampler) {
  std::cout << "Epoch " << epoch << ':' << std::endl;
  float train_avg_loss = ::process(model, trainer, train_sampler, true);
  std::cout << "  Train loss: " << train_avg_loss << std::endl;
  float dev_avg_loss = ::process(model, trainer, dev_sampler, false);
  std::cout << "  Dev loss: " << dev_avg_loss << std::endl;
  std::cout << "  Generating dev hyps ... " << std::flush;
  const std::vector<std::string> dev_hyps = ::infer_corpus(
      model, trg_vocab, dev_sampler);
  std::cout << "done." << std::endl;

  std::cout << "  Saving current model ... " << std::flush;
  ::save_all(
      ::get_model_dir(model_dir, epoch), model, trainer,
      train_avg_loss, dev_avg_loss, dev_hyps);
  std::cout << "done." << std::endl;
}

inline void train(
    unsigned last_epoch, unsigned num_epochs,
    const std::string &model_dir, ::AttentionEncoderDecoder &model,
    const ::Vocabulary &trg_vocab, primitiv::Trainer &trainer,
    ::Sampler &train_sampler, ::Sampler &dev_sampler) {
  for (unsigned i = 1; i <= num_epochs; ++i) {
    ::train_step(
        last_epoch + i,
        model_dir, model, trg_vocab, trainer, train_sampler, dev_sampler);
  }
}

#endif  // MYMT_NMT_UTILS_H_
