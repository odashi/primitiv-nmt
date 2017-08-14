#ifndef MYMT_ATTENTION_NMT_H_
#define MYMT_ATTENTION_NMT_H_

#include <primitiv/primitiv.h>

#include "lstm.h"
#include "utils.h"

// Encoder-decoder translation model with dot-attention.
class AttentionNMT {
  string name_;
  unsigned embed_size_;
  float dropout_rate_;
  primitiv::Parameter psrc_lookup_, ptrg_lookup_, pwhj_, pbj_, pwjy_, pby_;
  ::LSTM src_fw_lstm_, src_bw_lstm_, trg_lstm_;
  primitiv::Node trg_lookup_, whj_, bj_, wjy_, by_, concat_fb_, t_concat_fb_, feed_;

public:
  AttentionNMT(const string &name,
      unsigned src_vocab_size, unsigned trg_vocab_size,
      unsigned embed_size, unsigned hidden_size, float dropout_rate)
    : name_(name)
    , embed_size_(embed_size)
    , dropout_rate_(dropout_rate)
    , psrc_lookup_(name_ + ".src_lookup", {embed_size, src_vocab_size},
        primitiv::initializers::XavierUniform())
    , ptrg_lookup_(name_ + ".trg_lookup", {embed_size, trg_vocab_size},
        primitiv::initializers::XavierUniform())
    , pwhj_(name_ + ".whj", {embed_size, 2 * hidden_size},
        primitiv::initializers::XavierUniform())
    , pbj_(name_ + ".bj", {embed_size},
        primitiv::initializers::Constant(0))
    , pwjy_(name_ + ".wjy", {trg_vocab_size, embed_size},
        primitiv::initializers::XavierUniform())
    , pby_(name_ + ".by", {trg_vocab_size},
        primitiv::initializers::Constant(0))
    , src_fw_lstm_(name_ + ".src_fw_lstm", embed_size, hidden_size)
    , src_bw_lstm_(name_ + ".src_bw_lstm", embed_size, hidden_size)
    , trg_lstm_(name + ".trg_lstm", 2 * embed_size, hidden_size) {}

  // Loads all parameters.
  AttentionNMT(const string &name, const string &prefix)
    : name_(name)
    , psrc_lookup_(Parameter::load(prefix + name_ + ".src_lookup"))
    , ptrg_lookup_(Parameter::load(prefix + name_ + ".trg_lookup"))
    , pwhj_(Parameter::load(prefix + name_ + ".whj"))
    , pbj_(Parameter::load(prefix + name_ + ".bj"))
    , pwjy_(Parameter::load(prefix + name_ + ".wjy"))
    , pby_(Parameter::load(prefix + name_ + ".by"))
    , src_fw_lstm_(name_ + ".src_fw_lstm", prefix)
    , src_bw_lstm_(name_ + ".src_bw_lstm", prefix)
    , trg_lstm_(name_ + ".trg_lstm", prefix) {
      embed_size_ = pbj_.shape()[0];
      ifstream ifs;
      ::open_file(prefix + name_ + ".config", ifs);
      ifs >> dropout_rate_;
    }

  // Saves all parameters.
  void save(const string &prefix) const {
    psrc_lookup_.save(prefix + psrc_lookup_.name());
    ptrg_lookup_.save(prefix + ptrg_lookup_.name());
    pwhj_.save(prefix + pwhj_.name());
    pbj_.save(prefix + pbh_.name());
    pwjy_.save(prefix + pwjy_.name());
    pby_.save(prefix + pby_,name());
    src_fw_lstm_.save(prefix);
    src_bw_lstm_.save(prefix);
    trg_lstm_.save(prefix);
    ofstream ofs;
    ::open_file(prefix + name_ + ".config", ofs);
    ofs << dropout_rate_ << endl;
  }

  // Adds parameters to the trainer.
  void register_training(Trainer &trainer) {
    trainer.add_parameter(psrc_lookup_);
    trainer.add_parameter(ptrg_lookup_);
    trainer.add_parameter(pwhj_);
    trainer.add_parameter(pbj_);
    trainer.add_parameter(pwjy_);
    trainer.add_parameter(pby_);
    src_fw_lstm_.register_training(trainer);
    src_bw_lstm_.register_training(trainer);
    trg_lstm_.register_training(trainer);
  }

  // Encodes source sentences and prepare internal states.
  void encode(const vector<vector<unsigned>> &src_batch, bool train) {
    // Embedding lookup.
    const Node src_lookup = F::input(psrc_lookup_);
    vector<Node> e_list;
    for (const auto &x : src_batch) {
      e_list.emplace_back(
          F::dropout(F::pick(src_lookup, x, 1), dropout_rate_, train));
    }

    // Forward encoding.
    src_fw_lstm_.init();
    vector<Node> f_list;
    for (const auto &e : e_list) {
      f_list.emplace_back(
          F::dropout(src_fw_lstm_.forward(e), dropout_rate_, train));
    }

    // Backward encoding.
    src_bw_lstm_.init();
    vector<Node> b_list;
    for (auto it = e_list.rbegin(); it != e_list.rend(); ++it) {
      b_list.emplace_back(
          F::dropout(src_bw_lstm_.forward(*it), dropout_rate_, train));
    }
    reverse(begin(b_list), end(b_list));

    // Concatenates RNN states.
    vector<Node> fb_list;
    for (unsigned i = 0; i < src_batch.size(); ++i) {
      fb_list.emplace_back(f_list[i] + b_list[i]);
    }
    concat_fb_ = F::concat(fb_list, 1);
    t_concat_fb_ = F::transpose(concat_fb_);

    // Initializes decoder states.
    trg_lookup_ = F::input(ptrg_lookup_);
    whj_ = F::input(pwhj_);
    bj_ = F::input(pbj_);
    wjy_ = F::input(pwjy_);
    by_ = F::input(pby_);
    feed_ = F::zeros({embed_size_});
    trg_lstm_.init(
        src_fw_lstm_.get_c() + src_bw_lstm_.get_c(),
        src_fw_lstm_.get_h() + src_bw_lstm_.get_h());
  }

  // One step decoding.
  Node decode_step(const vector<unsigned> &trg_words, bool train) {
    Node e = F::pick(trg_lookup_, trg_words, 1);
    e = F::dropout(e, dropout_rate_, train);
    Node h = trg_lstm_.forward(F::concat({e, feed_}, 0));
    h = F::dropout(h, dropout_rate_, train);
    const Node atten_probs = F::softmax(F::matmul(t_concat_fb_, h), 0);
    const Node c = F::matmul(concat_fb_, atten_probs);
    feed_ = F::tanh(F::matmul(whj_, F::concat({h, c}, 0)) + bj_);
    return F::matmul(wjy_, feed_) + by_;
  }

  // Calculates the loss function over given target sentences.
  Node loss(const vector<vector<unsigned>> &trg_batch, bool train) {
    vector<Node> losses;
    for (unsigned i = 0; i < trg_batch.size() - 1; ++i) {
      Node y = decode_step(trg_batch[i], train);
      losses.emplace_back(F::softmax_cross_entropy(y, trg_batch[i + 1], 0));
    }
    return F::batch::mean(F::sum(losses));
  }
};

#endif  // MYMT_ATTENTION_NMT_H_
