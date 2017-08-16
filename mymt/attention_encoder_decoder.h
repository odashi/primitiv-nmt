#ifndef MYMT_ENCODER_DECODER_H_
#define MYMT_ENCODER_DECODER_H_

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include <primitiv/primitiv.h>

#include "lstm.h"
#include "utils.h"

class AttentionEncoderDecoder {
  std::string name_;
  unsigned src_vocab_size_, trg_vocab_size_;
  unsigned embed_size_, hidden_size_;
  float dropout_rate_;
  primitiv::Parameter pl_src_xe_, pl_trg_xe_;
  primitiv::Parameter pw_brd_fbd_, pb_brd_d_;
  primitiv::Parameter pw_att_fbh_, pw_att_dh_, pb_att_h_, pw_att_ha_;
  primitiv::Parameter pw_dec_cdj_, pb_dec_j_, pw_dec_jy_, pb_dec_y_;
  ::LSTM rnn_fw_, rnn_bw_, rnn_dec_;
  primitiv::Node fb_, fbh_, bcast_, j_;
  primitiv::Node l_trg_xe_;
  primitiv::Node w_att_dh_, b_att_h_, w_att_ha_;
  primitiv::Node w_dec_cdj_, b_dec_j_, w_dec_jy_, b_dec_y_;

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
    , pw_brd_fbd_(name_ + ".w_brd_fbd", {hidden_size_, 2 * hidden_size_},
        primitiv::initializers::XavierUniform())
    , pb_brd_d_(name_ + ".b_brd_d", {hidden_size_},
        primitiv::initializers::Constant(0))
    , pw_att_fbh_(name_ + ".w_att_fbh", {hidden_size_, 2 * hidden_size_},
        primitiv::initializers::XavierUniform())
    , pw_att_dh_(name_ + ".w_att_dh", {hidden_size_, hidden_size_},
        primitiv::initializers::XavierUniform())
    , pb_att_h_(name_ + ".b_att_h", {hidden_size_},
        primitiv::initializers::Constant(0))
    , pw_att_ha_(name_ + ".w_att_ha", {1, hidden_size_},
        primitiv::initializers::XavierUniform())
    , pw_dec_cdj_(name_ + ".w_dec_cdj", {embed_size_, 3 * hidden_size_},
        primitiv::initializers::XavierUniform())
    , pb_dec_j_(name_ + ".b_dec_j", {embed_size_},
        primitiv::initializers::Constant(0))
    , pw_dec_jy_(name_ + ".w_dec_jy", {trg_vocab_size_, embed_size_},
        primitiv::initializers::XavierUniform())
    , pb_dec_y_(name_ + ".b_dec_y", {trg_vocab_size_},
        primitiv::initializers::Constant(0))
    , rnn_fw_(name_ + ".rnn_fw", embed_size_, hidden_size_)
    , rnn_bw_(name_ + ".rnn_bw", embed_size_, hidden_size_)
    , rnn_dec_(name_ + ".rnn_dec", 2 * embed_size_, hidden_size_) {}

  // Loads all parameters.
  AttentionEncoderDecoder(const std::string &name, const std::string &prefix)
    : name_(name)
    , pl_src_xe_(primitiv::Parameter::load(prefix + name_ + ".l_src_xe"))
    , pl_trg_xe_(primitiv::Parameter::load(prefix + name_ + ".l_trg_xe"))
    , pw_brd_fbd_(primitiv::Parameter::load(prefix + name_ + ".w_brd_fbd"))
    , pb_brd_d_(primitiv::Parameter::load(prefix + name_ + ".b_brd_d"))
    , pw_att_fbh_(primitiv::Parameter::load(prefix + name_ + ".w_att_fbh"))
    , pw_att_dh_(primitiv::Parameter::load(prefix + name_ + ".w_att_dh"))
    , pb_att_h_(primitiv::Parameter::load(prefix + name_ + ".b_att_h"))
    , pw_att_ha_(primitiv::Parameter::load(prefix + name_ + ".w_att_ha"))
    , pw_dec_cdj_(primitiv::Parameter::load(prefix + name_ + ".w_dec_cdj"))
    , pb_dec_j_(primitiv::Parameter::load(prefix + name_ + ".b_dec_j"))
    , pw_dec_jy_(primitiv::Parameter::load(prefix + name_ + ".w_dec_jy"))
    , pb_dec_y_(primitiv::Parameter::load(prefix + name_ + ".b_dec_y"))
    , rnn_fw_(name_ + ".rnn_fw", prefix)
    , rnn_bw_(name_ + ".rnn_bw", prefix)
    , rnn_dec_(name_ + ".rnn_dec", prefix) {
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
    pw_brd_fbd_.save(prefix + pw_brd_fbd_.name());
    pb_brd_d_.save(prefix + pb_brd_d_.name());
    pw_att_fbh_.save(prefix + pw_att_fbh_.name());
    pw_att_dh_.save(prefix + pw_att_dh_.name());
    pb_att_h_.save(prefix + pb_att_h_.name());
    pw_att_ha_.save(prefix + pw_att_ha_.name());
    pw_dec_cdj_.save(prefix + pw_dec_cdj_.name());
    pb_dec_j_.save(prefix + pb_dec_j_.name());
    pw_dec_jy_.save(prefix + pw_dec_jy_.name());
    pb_dec_y_.save(prefix + pb_dec_y_.name());
    rnn_fw_.save(prefix);
    rnn_bw_.save(prefix);
    rnn_dec_.save(prefix);
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
    trainer.add_parameter(pw_brd_fbd_);
    trainer.add_parameter(pb_brd_d_);
    trainer.add_parameter(pw_att_fbh_);
    trainer.add_parameter(pw_att_dh_);
    trainer.add_parameter(pb_att_h_);
    trainer.add_parameter(pw_att_ha_);
    trainer.add_parameter(pw_dec_cdj_);
    trainer.add_parameter(pb_dec_j_);
    trainer.add_parameter(pw_dec_jy_);
    trainer.add_parameter(pb_dec_y_);
    rnn_fw_.register_training(trainer);
    rnn_bw_.register_training(trainer);
    rnn_dec_.register_training(trainer);
  }

  // Encodes source batch and initializes decoder states.
  void encode(
      const std::vector<std::vector<unsigned>> &src_batch, bool train) {
    namespace F = primitiv::node_ops;
    using primitiv::Node;
    const unsigned src_len = src_batch.size();

    // Source embedding
    const Node l_src_xe = F::input(pl_src_xe_);
    std::vector<Node> e_list;
    for (const auto &x : src_batch) {
      e_list.emplace_back(F::pick(l_src_xe, x, 1));
    }

    // Forward encoding
    rnn_fw_.init();
    std::vector<Node> f_list;
    for (const Node &e : e_list) {
      f_list.emplace_back(rnn_fw_.forward(e));
    }

    // Backward encoding
    rnn_bw_.init();
    std::vector<Node> b_list;
    for (auto it = e_list.rbegin(); it != e_list.rend(); ++it) {
      b_list.emplace_back(rnn_bw_.forward(*it));
    }
    std::reverse(b_list.begin(), b_list.end());

    // Initializing decoder states
    const Node w_brd_fbd = F::input(pw_brd_fbd_);
    const Node b_brd_d = F::input(pb_brd_d_);
    const Node last_fb = F::concat({rnn_fw_.get_c(), rnn_bw_.get_c()}, 0);
    rnn_dec_.init(F::matmul(w_brd_fbd, last_fb) + b_brd_d);

    // Making matrix for calculatin attention
    const Node w_att_fbh = F::input(pw_att_fbh_);
    std::vector<Node> fb_list;
    for (unsigned i = 0; i < src_len; ++i) {
      fb_list.emplace_back(F::concat({f_list[i], b_list[i]}, 0));
    }
    fb_ = F::concat(fb_list, 1);  // 2H x |src|
    fbh_ = F::matmul(w_att_fbh, fb_);  // H x |src|
    bcast_ = F::ones({1, src_len});
    j_ = F::zeros({embed_size_});

    // Other parameters
    l_trg_xe_ = F::input(pl_trg_xe_);
    w_att_dh_ = F::input(pw_att_dh_);
    b_att_h_ = F::input(pb_att_h_);
    w_att_ha_ = F::input(pw_att_ha_);
    w_dec_cdj_ = F::input(pw_dec_cdj_);
    b_dec_j_ = F::input(pb_dec_j_);
    w_dec_jy_ = F::input(pw_dec_jy_);
    b_dec_y_ = F::input(pb_dec_y_);
  }

  // One-step decoding
  primitiv::Node decode_step(
      const std::vector<unsigned> &trg_words, bool train) {
    namespace F = primitiv::node_ops;
    using primitiv::Node;

    const Node e = F::pick(l_trg_xe_, trg_words, 1);
    const Node d = rnn_dec_.forward(F::concat({e, j_}, 0));
    const Node dh = F::matmul(w_att_dh_, d) + b_att_h_;
    const Node h = F::tanh(fbh_ + F::matmul(dh, bcast_));  // H x |src|
    const Node a_logit = F::transpose(F::matmul(w_att_ha_, h));  // |src| x 1
    const Node a_prob = F::softmax(a_logit, 0);
    const Node c = F::matmul(fb_, a_prob);  // 2H
    j_ = F::tanh(F::matmul(w_dec_cdj_, F::concat({c, d}, 0)) + b_dec_j_);
    return F::matmul(w_dec_jy_, j_) + b_dec_y_;
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

#endif  // MYMT_ENCODER_DECODER_H_
