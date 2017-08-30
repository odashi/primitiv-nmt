#ifndef MYMT_ENCODER_DECODER_H_
#define MYMT_ENCODER_DECODER_H_

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include <primitiv/primitiv.h>

#include "affine.h"
#include "attention.h"
#include "lstm.h"
#include "utils.h"

class AttentionEncoderDecoder {
  std::string name_;
  unsigned src_vocab_size_, trg_vocab_size_;
  unsigned embed_size_, hidden_size_;
  float dropout_rate_;
  primitiv::Parameter pl_src_xe_, pl_trg_xe_;
  ::LSTM rnn_fw_, rnn_bw_, rnn_dec_;
  ::Attention att_;
  ::Affine aff_fbd_, aff_cdj_, aff_jy_;
  primitiv::Node d_, j_;
  primitiv::Node l_trg_xe_;

  AttentionEncoderDecoder(const AttentionEncoderDecoder &) = delete;
  AttentionEncoderDecoder &operator=(const AttentionEncoderDecoder &) = delete;

public:
  AttentionEncoderDecoder(const std::string &name,
      unsigned src_vocab_size, unsigned trg_vocab_size,
      unsigned embed_size, unsigned hidden_size,
      float dropout_rate)
    : name_(name)
    , src_vocab_size_(src_vocab_size)
    , trg_vocab_size_(trg_vocab_size)
    , embed_size_(embed_size)
    , hidden_size_(hidden_size)
    , dropout_rate_(dropout_rate)
    , pl_src_xe_(name_ + ".l_src_xe", {embed_size_, src_vocab_size_},
        primitiv::initializers::XavierUniform())
    , pl_trg_xe_(name_ + ".l_trg_xe", {embed_size_, trg_vocab_size_},
        primitiv::initializers::XavierUniform())
    , rnn_fw_(name_ + ".rnn_fw", embed_size_, hidden_size_, dropout_rate_)
    , rnn_bw_(name_ + ".rnn_bw", embed_size_, hidden_size_, dropout_rate_)
    , rnn_dec_(name_ + ".rnn_dec", 2 * embed_size_, hidden_size_, dropout_rate_)
    , att_(name_ + ".att", 2 * hidden_size_, hidden_size_, hidden_size_)
    , aff_fbd_(name_ + ".aff_fbd", 2 * hidden_size_, hidden_size_)
    , aff_cdj_(name_ + ".aff_cdj", 3 * hidden_size_, embed_size_)
    , aff_jy_(name_ + ".aff_jy", embed_size_, trg_vocab_size_) {}

  // Loads all parameters.
  AttentionEncoderDecoder(const std::string &name, const std::string &prefix)
    : name_(name)
    , pl_src_xe_(primitiv::Parameter::load(prefix + name_ + ".l_src_xe"))
    , pl_trg_xe_(primitiv::Parameter::load(prefix + name_ + ".l_trg_xe"))
    , rnn_fw_(name_ + ".rnn_fw", prefix)
    , rnn_bw_(name_ + ".rnn_bw", prefix)
    , rnn_dec_(name_ + ".rnn_dec", prefix)
    , att_(name_ + ".att", prefix)
    , aff_fbd_(name_ + ".aff_fbd", prefix)
    , aff_cdj_(name_ + ".aff_cdj", prefix)
    , aff_jy_(name_ + ".aff_jy", prefix) {
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
    pl_src_xe_.save(prefix + pl_src_xe_.name());
    pl_trg_xe_.save(prefix + pl_trg_xe_.name());
    rnn_fw_.save(prefix);
    rnn_bw_.save(prefix);
    rnn_dec_.save(prefix);
    att_.save(prefix);
    aff_fbd_.save(prefix);
    aff_cdj_.save(prefix);
    aff_jy_.save(prefix);
    std::ofstream ofs;
    ::open_file(prefix + name_ + ".config", ofs);
    ofs << src_vocab_size_ << std::endl;
    ofs << trg_vocab_size_ << std::endl;
    ofs << embed_size_ << std::endl;
    ofs << hidden_size_ << std::endl;
    ofs << dropout_rate_ << std::endl;
  }

  // Adds parameters to the trainer.
  void register_training(primitiv::Trainer &trainer) {
    trainer.add_parameter(pl_src_xe_);
    trainer.add_parameter(pl_trg_xe_);
    rnn_fw_.register_training(trainer);
    rnn_bw_.register_training(trainer);
    rnn_dec_.register_training(trainer);
    att_.register_training(trainer);
    aff_fbd_.register_training(trainer);
    aff_cdj_.register_training(trainer);
    aff_jy_.register_training(trainer);
  }

  // Encodes source batch and initializes decoder states.
  void encode(const std::vector<std::vector<unsigned>> &src_batch, bool train) {
    namespace F = primitiv::node_ops;

    const unsigned src_len = src_batch.size();

    // Source embedding
    const auto l_src_xe = F::input(pl_src_xe_);
    std::vector<primitiv::Node> e_list;
    for (const auto &x : src_batch) {
      e_list.emplace_back(F::pick(l_src_xe, x, 1));
    }

    // Forward encoding
    rnn_fw_.init(primitiv::Node(), primitiv::Node(), train);
    std::vector<primitiv::Node> f_list;
    for (const auto &e : e_list) {
      f_list.emplace_back(rnn_fw_.forward(e, train));
    }

    // Backward encoding
    rnn_bw_.init(primitiv::Node(), primitiv::Node(), train);
    std::vector<primitiv::Node> b_list;
    for (auto it = e_list.rbegin(); it != e_list.rend(); ++it) {
      b_list.emplace_back(rnn_bw_.forward(*it, train));
    }
    std::reverse(b_list.begin(), b_list.end());

    // Initializing decoder states
    aff_fbd_.init();
    aff_cdj_.init();
    aff_jy_.init();
    const auto last_fb = F::concat({rnn_fw_.get_c(), rnn_bw_.get_c()}, 0);
    rnn_dec_.init(aff_fbd_.forward(last_fb), primitiv::Node(), train);

    // Making matrix for calculating attention
    std::vector<primitiv::Node> fb_list;
    for (unsigned i = 0; i < src_len; ++i) {
      fb_list.emplace_back(F::concat({f_list[i], b_list[i]}, 0));
    }
    att_.init(fb_list);

    // Initial output embedding (feeding) vector.
    j_ = F::zeros({embed_size_});

    // Other parameters
    l_trg_xe_ = F::input(pl_trg_xe_);
  }

  // Calculates next attention probabilities
  primitiv::Node decode_atten(
      const std::vector<unsigned> &trg_words, bool train) {
    namespace F = primitiv::node_ops;

    const auto e = F::pick(l_trg_xe_, trg_words, 1);
    d_ = rnn_dec_.forward(F::concat({e, j_}, 0), train);
    return att_.get_probs(d_);
  }

  // Calculates next words
  primitiv::Node decode_word(const primitiv::Node &att_probs, bool train) {
    namespace F = primitiv::node_ops;

    const auto c = att_.get_context(att_probs);
    j_ = F::tanh(aff_cdj_.forward(F::concat({c, d_}, 0)));
    return aff_jy_.forward(j_);
  }

  // Calculates the loss function.
  primitiv::Node loss(
      const std::vector<std::vector<unsigned>> &trg_batch, bool train) {
    namespace F = primitiv::node_ops;

    std::vector<primitiv::Node> losses;
    for (unsigned i = 0; i < trg_batch.size() - 1; ++i) {
      const auto att_probs = decode_atten(trg_batch[i], train);
      const auto y = decode_word(att_probs, train);
      losses.emplace_back(F::softmax_cross_entropy(y, trg_batch[i + 1], 0));
    }
    return F::batch::mean(F::sum(losses));
  }
};

#endif  // MYMT_ENCODER_DECODER_H_
