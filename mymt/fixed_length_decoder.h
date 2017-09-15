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

template<typename Var>
class FixedLengthDecoder {
  std::string name_;
  unsigned nv_src_, nv_trg_;
  unsigned ne_, nh_;
  float dr_;
  primitiv::Parameter pl_src_xe_, pl_trg_xe_;
  ::LSTM<Var> rnn_enc_fw_, rnn_enc_bw_, rnn_dec_fw_, rnn_dec_bw_;
  ::Attention<Var> att_;
  ::Affine<Var> aff_ed_, aff_cdj_, aff_jy_;
  Var l_trg_xe_, dec_fw_c0_, dec_bw_c0_;

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
        primitiv::initializers::Uniform(-0.1, 0.1))
    , pl_trg_xe_(name_ + ".l_trg_xe", {ne_, nv_trg_},
        primitiv::initializers::Uniform(-0.1, 0.1))
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

    const unsigned src_len = src_batch.size();
    const Var invalid;

    // Source embedding
    const auto l_src_xe = F::input<Var>(pl_src_xe_);
    std::vector<Var> e_list;
    for (const auto &x : src_batch) {
      e_list.emplace_back(F::pick(l_src_xe, x, 1));
    }

    // Forward encoding
    rnn_enc_fw_.init(invalid, invalid, train);
    std::vector<Var> f_list;
    for (const auto &e : e_list) {
      f_list.emplace_back(rnn_enc_fw_.forward(e, train));
    }

    // Backward encoding
    rnn_enc_bw_.init(invalid, invalid, train);
    std::vector<Var> b_list;
    for (auto it = e_list.rbegin(); it != e_list.rend(); ++it) {
      b_list.emplace_back(rnn_enc_bw_.forward(*it, train));
    }
    std::reverse(b_list.begin(), b_list.end());

    // Initializing fixed states
    aff_ed_.init();
    aff_cdj_.init();
    aff_jy_.init();
    const auto last_enc_fb = F::concat(
        {rnn_enc_fw_.get_c(), rnn_enc_bw_.get_c()}, 0);
    const auto init_dec_fb = aff_ed_.forward(last_enc_fb);
    dec_fw_c0_ = F::slice(init_dec_fb, 0, 0, nh_);
    dec_bw_c0_ = F::slice(init_dec_fb, 0, nh_, 2 * nh_);

    // Making matrix for calculating attention (ignoring <bos> and <eos>)
    std::vector<Var> fb_list;
    for (unsigned i = 1; i < src_len - 1; ++i) {
      fb_list.emplace_back(F::concat({f_list[i], b_list[i]}, 0));
    }
    att_.init(fb_list);

    // Other parameters.
    l_trg_xe_ = F::input<Var>(pl_trg_xe_);
  }

  struct SamplingResult {
    unsigned org_id, new_id;
    float org_score, new_score;
  };

  // Calculates a sample.
  SamplingResult sample(
      const std::vector<std::vector<unsigned>> &trg_batch,
      unsigned pos) {
    namespace F = primitiv::operators;

    const unsigned trg_len = trg_batch.size();
    if (pos >= trg_len - 2) throw std::runtime_error("invalid pos");

    // Initializing decoder stetes
    const Var invalid;
    rnn_dec_fw_.init(dec_fw_c0_, invalid, false);
    rnn_dec_bw_.init(dec_bw_c0_, invalid, false);

    // Target embedding
    std::vector<Var> e_list;
    for (const auto &x : trg_batch) {
      e_list.emplace_back(F::pick(l_trg_xe_, x, 1));
    }

    // Forward RNN
    std::vector<Var> f_list;
    for (const auto &e : e_list) {
      f_list.emplace_back(rnn_dec_fw_.forward(e, false));
    }

    // Backward RNN
    std::vector<Var> b_list;
    for (auto it = e_list.rbegin(); it != e_list.rend(); ++it) {
      b_list.emplace_back(rnn_dec_bw_.forward(*it, false));
    }
    std::reverse(b_list.begin(), b_list.end());

    // Calculates positional probs.
    const auto d = F::concat({f_list[pos], b_list[pos + 2]}, 0);
    const auto a_probs = att_.get_probs(d);
    const auto c = att_.get_context(a_probs);
    const auto j = F::tanh(aff_cdj_.forward(F::concat({c, d}, 0)));
    const auto y = aff_jy_.forward(j);
    const auto log_probs = F::log_softmax(y, 0);
    const auto log_probs_v = log_probs.to_vector();

    // Sample a word.
    const auto noise = F::random::gumbel<Var>(log_probs.shape(), 0, 1);
    const unsigned sample = ::argmax((log_probs + noise).to_vector());

    return SamplingResult {
      trg_batch[pos + 1][0], sample,
      log_probs_v[trg_batch[pos + 1][0]], log_probs_v[sample],
    };
  }

  // Calculates loss function.
  Var loss(const std::vector<std::vector<unsigned>> &trg_batch, bool train) {
    namespace F = primitiv::operators;

    const unsigned trg_len = trg_batch.size();

    // Initializing decoder stetes
    const Var invalid;
    rnn_dec_fw_.init(dec_fw_c0_, invalid, train);
    rnn_dec_bw_.init(dec_bw_c0_, invalid, train);

    // Target embedding
    std::vector<Var> e_list;
    for (const auto &x : trg_batch) {
      e_list.emplace_back(F::pick(l_trg_xe_, x, 1));
    }

    // Forward RNN
    std::vector<Var> f_list;
    for (const auto &e : e_list) {
      f_list.emplace_back(rnn_dec_fw_.forward(e, train));
    }

    // Backward RNN
    std::vector<Var> b_list;
    for (auto it = e_list.rbegin(); it != e_list.rend(); ++it) {
      b_list.emplace_back(rnn_dec_bw_.forward(*it, train));
    }
    std::reverse(b_list.begin(), b_list.end());

    // Calculates losses (ignoring <bos> and <eos>)
    std::vector<Var> losses;
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
