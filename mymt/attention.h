#ifndef MYMT_ATTENTION_H_
#define MYMT_ATTENTION_H_

#include <fstream>
#include <string>
#include <vector>

#include <primitiv/primitiv.h>

#include "utils.h"

// Multilayer perceptron-based attention
class Attention {
  std::string name_;
  unsigned ne_, nd_, nh_;
  primitiv::Parameter pweh_, pwdh_, pbh_, pwha_;
  primitiv::Node e_mat_, eh_mat_, wdh_, bh_, wha_;

  Attention(const Attention &) = delete;
  Attention &operator=(const Attention &) = delete;

public:
  Attention(
    const std::string &name,
    unsigned encoder_size, unsigned decoder_size, unsigned hidden_size)
    : name_(name)
    , ne_(encoder_size)
    , nd_(decoder_size)
    , nh_(hidden_size)
    , pweh_(name_ + ".weh", {nh_, ne_}, primitiv::initializers::XavierUniform())
    , pwdh_(name_ + ".wdh", {nh_, nd_}, primitiv::initializers::XavierUniform())
    , pbh_(name_ + ".bh", {nh_}, primitiv::initializers::Constant(0))
    , pwha_(name_ + ".wha", {1, nh_}, primitiv::initializers::XavierUniform())
  {}

  Attention(const std::string &name, const std::string &prefix)
    : name_(name)
    , pweh_(primitiv::Parameter::load(prefix + name_ + ".weh"))
    , pwdh_(primitiv::Parameter::load(prefix + name_ + ".wdh"))
    , pbh_(primitiv::Parameter::load(prefix + name_ + ".bh"))
    , pwha_(primitiv::Parameter::load(prefix + name_ + ".wha")) {
      std::ifstream ifs;
      ::open_file(prefix + name_ + ".config", ifs);
      ifs >> ne_ >> nd_ >> nh_;
    }

  void save(const std::string &prefix) const {
    pweh_.save(prefix + pweh_.name());
    pwdh_.save(prefix + pwdh_.name());
    pbh_.save(prefix + pbh_.name());
    pwha_.save(prefix + pwha_.name());
    std::ofstream ofs;
    ::open_file(prefix + name_ + ".config", ofs);
    ofs << ne_ << std::endl;
    ofs << nd_ << std::endl;
    ofs << nh_ << std::endl;
  }

  void register_training(primitiv::Trainer &trainer) {
    trainer.add_parameter(pweh_);
    trainer.add_parameter(pwdh_);
    trainer.add_parameter(pbh_);
    trainer.add_parameter(pwha_);
  }

  void init(const std::vector<primitiv::Node> &encoder_states) {
    namespace F = primitiv::node_ops;
    using primitiv::Node;

    const Node weh = F::input(pweh_);
    e_mat_ = F::concat(encoder_states, 1);  // ne_ x len
    eh_mat_ = F::matmul(weh, e_mat_);  // nh_ x len

    wdh_ = F::input(pwdh_);
    bh_ = F::input(pbh_);
    wha_ = F::input(pwha_);
  }

  primitiv::Node get_probs(const primitiv::Node &decoder_state) {
    namespace F = primitiv::node_ops;
    using primitiv::Node;

    const Node dh = F::matmul(wdh_, decoder_state) + bh_;  // nh_ x 1
    const Node dh_bcast = F::broadcast(dh, 1, eh_mat_.shape()[1]);  // nh_ x len
    const Node h = F::tanh(eh_mat_ + dh_bcast);
    const Node a = F::transpose(F::matmul(wha_, h));  // len x 1
    return F::softmax(a, 0);
  }

  primitiv::Node get_context(const primitiv::Node &att_probs) {
    return primitiv::node_ops::matmul(e_mat_, att_probs);
  }
};

#endif  // MYMT_ATTENTION_H_
