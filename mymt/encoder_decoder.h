#ifndef MYMT_ENCODER_DECODER_H_
#define MYMT_ENCODER_DECODER_H_

#include <fstream>
#include <string>
#include <vector>

#include <primitiv/primitiv.h>

#include "lstm.h"
#include "utils.h"

class EncoderDecoder {
  std::string name_;
  unsigned src_vocab_size_, trg_vocab_size_;
  unsigned embed_size_, hidden_size_;
  float dropout_rate_;
  primitiv::Parameter enc_plxe_, dec_plxe_, dec_pwhy_, dec_pby_;
  ::LSTM enc_rnn_, dec_rnn_;
  primitiv::Node dec_lxe_, dec_why_, dec_by_;

  EncoderDecoder(const EncoderDecoder &) = delete;
  EncoderDecoder &operator=(const EncoderDecoder &) = delete;

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
    ofs << src_vocab_size_ << std::endl;
    ofs << trg_vocab_size_ << std::endl;
    ofs << embed_size_ << std::endl;
    ofs << hidden_size_ << std::endl;
    ofs << dropout_rate_ << std::endl;
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

#endif  // MYMT_ENCODER_DECODER_H_
