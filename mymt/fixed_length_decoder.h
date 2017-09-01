#ifndef MYMT_FIXED_LENGTH_DECODER_H_
#define MYMT_FIXED_LENGTH_DECODER_H_

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include <primitiv/primitiv.h>

#include "affine.h"
#include "attention.h"
#include "lstm.h"
#include "utils.h"

class FixedLengthDecoder {
  std::string name_;
  unsigned nv_src_, nv_trg_;
  unsigned ne_, nh_;
  float dr_;
  primitiv::Parameter pl_src_xe_, pl_trg_xe_;
  ::LSTM rnn_enc_fw_, rnn_enc_bw_, rnn_dec_fw_, rnn_dec_bw_;
  ::Attention att_;
  ::Affine aff_ed_, aff_cdj_, aff_jy_;
  primitiv::Node l_trg_xe_;

  FixedLengthDecoder(const FixedLengthDecoder &) = delete;
  FixedLengthDecoder &operator=(const FixedLengthDecoder &) = delete;

public:
  FixedLengthDecoder(const std::string &name,
      unsigned src_vocab_size, unsigned trg_vocab_size,
      unsigned embed_size, unsigned hidden_size,
      float dropout_rate)
    : name_(name)
    , nv_src_(src_vocab_size)
    , nv_trg_(trg_vocab_size)
    , ne_(embed_size)
    , nh_(hidden_size)
    , dr_(dropout_rate)
    , pl_src_xe_(name_ + ".l_src_xe", {ne_, nv_src_},
        primitiv::initializers::XavierUniform())
    , pl_trg_xe_(name_ + ".l_trg_xe", {ne_, nv_trg_},
        primitiv::initializers::XavierUniform())
    , rnn_enc_fw_(name_ + ".rnn_enc_fw", ne_, nh_, dr_)
    , rnn_enc_bw_(name_ + ".rnn_enc_bw", ne_, nh_, dr_)
    , rnn_dec_fw_(name_ + ".rnn_dec_fw", ne_, nh_, dr_)
    , rnn_dec_bw_(name_ + ".rnn_dec_bw", ne_, nh_, dr_)
    , att_(name_ + ".att", 2 * nh_, 2 * nh_, nh_)
    , aff_ed_(name_ + ".aff_ed", 2 * nh_, 2 * nh_)
    , aff_cdj_(name_ + ".aff_cdj", 4 * nh_, ne_)
    , aff_jy_(name_ + ".aff_jy", ne_, nv_trg_) {}

  // Loads all parameters.
  FixedLengthDecoder(const std::string &name, const std::string &prefix)
    : name_(name)
    , pl_src_xe_(primitiv::Parameter::load(prefix + name_ + ".l_src_xe"))
    , pl_trg_xe_(primitiv::Parameter::load(prefix + name_ + ".l_trg_xe"))
    , rnn_enc_fw_(name_ + ".rnn_enc_fw", prefix)
    , rnn_enc_bw_(name_ + ".rnn_enc_bw", prefix)
    , rnn_dec_fw_(name_ + ".rnn_dec_fw", prefix)
    , rnn_dec_bw_(name_ + ".rnn_dec_bw", prefix)
    , att_(name_ + ".att", prefix)
    , aff_ed_(name_ + ".aff_ed", prefix)
    , aff_cdj_(name_ + ".aff_cdj", prefix)
    , aff_jy_(name_ + ".aff_jy", prefix) {
      std::ifstream ifs;
      ::open_file(prefix + name_ + ".config", ifs);
      ifs >> nv_src_;
      ifs >> nv_trg_;
      ifs >> ne_;
      ifs >> nh_;
      ifs >> dr_;
    }

  // Saves all parameters.
  void save(const std::string &prefix) const {
    pl_src_xe_.save(prefix + pl_src_xe_.name());
    pl_trg_xe_.save(prefix + pl_trg_xe_.name());
    rnn_enc_fw_.save(prefix);
    rnn_enc_bw_.save(prefix);
    rnn_dec_fw_.save(prefix);
    rnn_dec_bw_.save(prefix);
    att_.save(prefix);
    aff_ed_.save(prefix);
    aff_cdj_.save(prefix);
    aff_jy_.save(prefix);
    std::ofstream ofs;
    ::open_file(prefix + name_ + ".config", ofs);
    ofs << nv_src_ << std::endl;
    ofs << nv_trg_ << std::endl;
    ofs << ne_ << std::endl;
    ofs << nh_ << std::endl;
    ofs << dr_ << std::endl;
  }

  // Adds parameters to the trainer.
  void register_training(primitiv::Trainer &trainer) {
    trainer.add_parameter(pl_src_xe_);
    trainer.add_parameter(pl_trg_xe_);
    rnn_enc_fw_.register_training(trainer);
    rnn_enc_bw_.register_training(trainer);
    rnn_dec_fw_.register_training(trainer);
    rnn_dec_bw_.register_training(trainer);
    att_.register_training(trainer);
    aff_ed_.register_training(trainer);
    aff_cdj_.register_training(trainer);
    aff_jy_.register_training(trainer);
  }

  // Encodes source batch and initializes decoder states.
  void encode(const std::vector<std::vector<unsigned>> &src_batch, bool train) {
    namespace F = primitiv::operators;
    using primitiv::Node;

    const unsigned src_len = src_batch.size();
    const Node invalid;

    // Source embedding
    const auto l_src_xe = F::input<Node>(pl_src_xe_);
    std::vector<Node> e_list;
    for (const auto &x : src_batch) {
      e_list.emplace_back(F::pick(l_src_xe, x, 1));
    }

    // Forward encoding
    rnn_enc_fw_.init(invalid, invalid, train);
    std::vector<Node> f_list;
    for (const auto &e : e_list) {
      f_list.emplace_back(rnn_enc_fw_.forward(e, train));
    }

    // Backward encoding
    rnn_enc_bw_.init(invalid, invalid, train);
    std::vector<Node> b_list;
    for (auto it = e_list.rbegin(); it != e_list.rend(); ++it) {
      b_list.emplace_back(rnn_enc_bw_.forward(*it, train));
    }
    std::reverse(b_list.begin(), b_list.end());

    // Initializing decoder states
    aff_ed_.init();
    aff_cdj_.init();
    aff_jy_.init();
    const auto last_enc_fb = F::concat(
        {rnn_enc_fw_.get_c(), rnn_enc_bw_.get_c()}, 0);
    const auto init_dec_fb = aff_ed_.forward(last_enc_fb);
    rnn_dec_fw_.init(F::slice(init_dec_fb, 0, 0, nh_), invalid, train);
    rnn_dec_bw_.init(F::slice(init_dec_fb, 0, nh_, 2 * nh_), invalid, train);

    // Making matrix for calculating attention (ignoring <bos> and <eos>)
    std::vector<Node> fb_list;
    for (unsigned i = 1; i < src_len - 1; ++i) {
      fb_list.emplace_back(F::concat({f_list[i], b_list[i]}, 0));
    }
    att_.init(fb_list);

    // Other parameters.
    l_trg_xe_ = F::input<Node>(pl_trg_xe_);
  }

  // Calculates loss function.
  primitiv::Node loss(
      const std::vector<std::vector<unsigned>> &trg_batch, bool train) {
    namespace F = primitiv::operators;
    using primitiv::Node;

    const unsigned trg_len = trg_batch.size();

    // Target embedding
    std::vector<Node> e_list;
    for (const auto &x : trg_batch) {
      e_list.emplace_back(F::pick(l_trg_xe_, x, 1));
    }

    // Forward RNN
    std::vector<Node> f_list;
    for (const auto &e : e_list) {
      f_list.emplace_back(rnn_dec_fw_.forward(e, train));
    }

    // Backward RNN
    std::vector<Node> b_list;
    for (auto it = e_list.rbegin(); it != e_list.rend(); ++it) {
      b_list.emplace_back(rnn_dec_bw_.forward(*it, train));
    }
    std::reverse(b_list.begin(), b_list.end());

    // Calculates losses (ignoring <bos> and <eos>)
    std::vector<Node> losses;
    for (unsigned i = 1; i < trg_len - 1; ++i) {
      const auto d = F::concat({f_list[i - 1], b_list[i + 1]}, 0);
      const auto a_probs = att_.get_probs(d);
      const auto c = att_.get_context(a_probs);
      const auto j = F::tanh(aff_cdj_.forward(F::concat({c, d}, 0)));
      const auto y = aff_jy_.forward(j);
      losses.emplace_back(F::softmax_cross_entropy(y, trg_batch[i], 0));
    }
    return F::batch::mean(F::sum(losses));
  }
};

#endif  // MYMT_FIXED_LENGTH_DECODER_H_
