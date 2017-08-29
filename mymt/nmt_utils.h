#ifndef MYMT_NMT_UTILS_H_
#define MYMT_NMT_UTILS_H_

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <primitiv/primitiv.h>

#include "attention_encoder_decoder.h"
#include "sampler.h"
#include "utils.h"
#include "vocabulary.h"

struct Result {
  std::vector<unsigned> word_ids;
  std::vector<std::vector<float>> atten_probs;
};

inline ::Result infer_sentence(
    ::AttentionEncoderDecoder &model,
    unsigned bos_id, unsigned eos_id,
    const std::vector<std::vector<unsigned>> &src_batch,
    unsigned limit) {
  namespace F = primitiv::node_ops;

  // Initialize the model
  primitiv::Graph g;
  primitiv::Graph::set_default_graph(g);
  model.encode(src_batch, false);

  ::Result ret { {bos_id}, {} };

  // Decode
  while (ret.word_ids.back() != eos_id) {
    const std::vector<unsigned> prev {ret.word_ids.back()};
    const auto a_probs = model.decode_atten(prev, false);
    ret.atten_probs.emplace_back(g.forward(a_probs).to_vector());

    const auto scores = model.decode_word(a_probs, false);
    ret.word_ids.emplace_back(::argmax(g.forward(scores).to_vector()));

    if (ret.word_ids.size() == limit + 1) {
      ret.word_ids.emplace_back(eos_id);
      break;
    }
  }

  return ret;
}

inline ::Result infer_sentence_ensemble(
    std::vector<std::unique_ptr<primitiv::Device>> &devs,
    std::vector<std::unique_ptr<::AttentionEncoderDecoder>> &models,
    unsigned bos_id, unsigned eos_id,
    const std::vector<std::vector<unsigned>> &src_batch,
    unsigned limit) {
  namespace F = primitiv::node_ops;

  // Initialize the model
  primitiv::Graph g;
  primitiv::Graph::set_default_graph(g);
  for (unsigned i = 0; i < models.size(); ++i) {
    primitiv::Device::set_default_device(*devs[i % devs.size()]);
    models[i]->encode(src_batch, false);
  }

  ::Result ret { {bos_id}, {} };

  // Decode
  while (ret.word_ids.back() != eos_id) {
    std::vector<primitiv::Node> a_probs_list;
    std::vector<primitiv::Node> scores_list;
    const std::vector<unsigned> prev {ret.word_ids.back()};

    for (unsigned i = 0; i < models.size(); ++i) {
      primitiv::Device::set_default_device(*devs[i % devs.size()]);
      const auto a_probs = models[i]->decode_atten(prev, false);
      a_probs_list.emplace_back(F::copy(a_probs, *devs[0]));

      const auto scores = models[i]->decode_word(a_probs, false);
      scores_list.emplace_back(F::copy(scores, *devs[0]));
    }

    const auto a_probs_mean = F::mean(a_probs_list);
    const auto scores_sum = F::sum(scores_list);

    ret.atten_probs.emplace_back(g.forward(a_probs_mean).to_vector());
    ret.word_ids.emplace_back(::argmax(g.forward(scores_sum).to_vector()));

    if (ret.word_ids.size() == limit + 1) {
      ret.word_ids.emplace_back(eos_id);
      break;
    }
  }

  return ret;
}

inline ::Result infer_sentence_ensemble2(
    std::vector<std::unique_ptr<primitiv::Device>> &devs,
    std::vector<std::unique_ptr<::AttentionEncoderDecoder>> &models,
    unsigned bos_id, unsigned eos_id,
    const std::vector<std::vector<unsigned>> &src_batch,
    unsigned limit) {
  namespace F = primitiv::node_ops;

  // Initialize the model
  primitiv::Graph g;
  primitiv::Graph::set_default_graph(g);
  for (unsigned i = 0; i < models.size(); ++i) {
    primitiv::Device::set_default_device(*devs[i % devs.size()]);
    models[i]->encode(src_batch, false);
  }

  ::Result ret { {bos_id}, {} };

  // Decode
  while (ret.word_ids.back() != eos_id) {
    const std::vector<unsigned> prev {ret.word_ids.back()};

    std::vector<primitiv::Node> a_probs_list;
    for (unsigned i = 0; i < models.size(); ++i) {
      primitiv::Device::set_default_device(*devs[i % devs.size()]);
      const auto a_probs = models[i]->decode_atten(prev, false);
      a_probs_list.emplace_back(F::copy(a_probs, *devs[0]));
    }
    const auto a_probs_mean = F::mean(a_probs_list);

    std::vector<primitiv::Node> scores_list;
    for (unsigned i = 0; i < models.size(); ++i) {
      primitiv::Device::set_default_device(*devs[i % devs.size()]);
      const auto scores = models[i]->decode_word(F::copy(a_probs_mean), false);
      scores_list.emplace_back(F::copy(scores, *devs[0]));
    }
    const auto scores_sum = F::sum(scores_list);

    ret.atten_probs.emplace_back(g.forward(a_probs_mean).to_vector());
    ret.word_ids.emplace_back(::argmax(g.forward(scores_sum).to_vector()));

    if (ret.word_ids.size() == limit + 1) {
      ret.word_ids.emplace_back(eos_id);
      break;
    }
  }

  return ret;
}

inline std::string make_hyp_str(
    const ::Result &ret, const ::Vocabulary &trg_vocab) {
  std::string hyp_str;
  for (unsigned i = 1; i < ret.word_ids.size() - 1; ++i) {
    if (i > 1) hyp_str += ' ';
    hyp_str += trg_vocab.itos(ret.word_ids[i]);
  }
  return hyp_str;
}

class NMTTrainer {
  const std::string model_dir_;
  const ::Vocabulary &src_vocab_;
  const ::Vocabulary &trg_vocab_;
  ::AttentionEncoderDecoder &model_;
  primitiv::Trainer &opt_;
  ::Sampler &train_sampler_;
  ::Sampler &dev_sampler_;
  unsigned epoch_;
  float best_dev_avg_loss_;

  float process(::Sampler &sampler, bool train) {
    unsigned num_sents = 0;
    unsigned num_labels = 0;
    float accum_loss = 0;
    const unsigned total_sents = sampler.num_sentences();
    sampler.reset();

    while (sampler.has_next()) {
      const Batch batch = sampler.next();
      const unsigned batch_size = batch.source[0].size();

      primitiv::Graph g;
      primitiv::Graph::set_default_graph(g);
      model_.encode(batch.source, train);
      const auto loss = model_.loss(batch.target, train);
      accum_loss += g.forward(loss).to_vector()[0] * batch_size;

      if (train) {
        opt_.reset_gradients();
        g.backward(loss);
        opt_.update();
      }

      num_sents += batch_size;
      num_labels += batch_size * (batch.target.size() - 1);
      std::cout << num_sents << '/' << total_sents << '\r' << std::flush;
    }

    return accum_loss / num_labels;
  }

  std::vector<std::string> infer_corpus(::Sampler &sampler) {
    std::vector<std::string> hyps;
    sampler.reset();

    const unsigned bos_id = trg_vocab_.stoi("<bos>");
    const unsigned eos_id = trg_vocab_.stoi("<eos>");

    while (sampler.has_next()) {
      const Batch batch = sampler.next();
      const unsigned batch_size = batch.source[0].size();
      if (batch_size != 1) {
        throw std::runtime_error(
            "inference is allowed only for each one sentence.");
      }

      const ::Result ret = ::infer_sentence(
          model_, bos_id, eos_id, batch.source, 64);
      hyps.emplace_back(::make_hyp_str(ret, trg_vocab_));
    }

    return hyps;
  }

public:
  NMTTrainer(
      const std::string &model_dir,
      const ::Vocabulary &src_vocab,
      const ::Vocabulary &trg_vocab,
      ::AttentionEncoderDecoder &model,
      primitiv::Trainer &trainer,
      ::Sampler &train_sampler,
      ::Sampler &dev_sampler,
      unsigned epoch)
    : model_dir_(model_dir), src_vocab_(src_vocab), trg_vocab_(trg_vocab)
    , model_(model), opt_(trainer)
    , train_sampler_(train_sampler), dev_sampler_(dev_sampler)
    , epoch_(epoch)
    , best_dev_avg_loss_(
        ::load_value<float>(model_dir + "/best.dev_avg_loss")) {}

  void save(
      float train_avg_loss,
      float dev_avg_loss,
      const std::vector<std::string> &dev_hyps) {
    const std::string subdir = ::get_model_dir(model_dir_, epoch_);
    ::make_directory(subdir);
    model_.save(subdir + "/model.");
    opt_.save(subdir + "/trainer");
    ::save_value(subdir + "/train.avg_loss", train_avg_loss);
    ::save_value(subdir + "/dev.avg_loss", dev_avg_loss);
    ::save_strings(subdir + "/dev.hyp", dev_hyps);
  }

  void train() {
    std::cout << "Epoch " << ++epoch_ << ':' << std::endl;
    std::cout << "  Learning rate decay: "
              << opt_.get_learning_rate_scaling() << std::endl;

    float train_avg_loss = process(train_sampler_, true);
    std::cout << "  Train loss: " << train_avg_loss << std::endl;

    float dev_avg_loss = process(dev_sampler_, false);
    std::cout << "  Dev loss: " << dev_avg_loss << std::endl;

    if (dev_avg_loss < best_dev_avg_loss_) {
      std::cout << "    Best!" << std::endl;
      best_dev_avg_loss_ = dev_avg_loss;
      ::save_value(model_dir_ + "/best.epoch", epoch_);
      ::save_value(model_dir_ + "/best.dev_avg_loss", best_dev_avg_loss_);
    } else {
      const float prev_lr_decay = opt_.get_learning_rate_scaling();
      opt_.set_learning_rate_scaling(.5f * prev_lr_decay);
    }

    std::cout << "  Generating dev hyps ... " << std::flush;
    const std::vector<std::string> dev_hyps = infer_corpus(dev_sampler_);
    std::cout << "done." << std::endl;

    std::cout << "  Saving current model ... " << std::flush;
    save(train_avg_loss, dev_avg_loss, dev_hyps);
    std::cout << "done." << std::endl;
  }
};

#endif  // MYMT_NMT_UTILS_H_
