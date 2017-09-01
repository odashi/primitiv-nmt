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
  // New attention object.
  Attention(
    const std::string &name,
    unsigned encoder_size, unsigned decoder_size, unsigned hidden_size)
    : name_(name)
    , ne_(encoder_size)
    , nd_(decoder_size)
    , nh_(hidden_size)
    , pweh_(name_ + ".w_eh", {nh_, ne_},
        primitiv::initializers::XavierUniform())
    , pwdh_(name_ + ".w_dh", {nh_, nd_},
        primitiv::initializers::XavierUniform())
    , pbh_(name_ + ".b_h", {nh_},
        primitiv::initializers::Constant(0))
    , pwha_(name_ + ".w_ha", {1, nh_},
        primitiv::initializers::XavierUniform())
  {}

  // Loads parameters from files.
  Attention(const std::string &name, const std::string &prefix)
    : name_(name)
    , pweh_(primitiv::Parameter::load(prefix + name_ + ".w_eh"))
    , pwdh_(primitiv::Parameter::load(prefix + name_ + ".w_dh"))
    , pbh_(primitiv::Parameter::load(prefix + name_ + ".b_h"))
    , pwha_(primitiv::Parameter::load(prefix + name_ + ".w_ha")) {
      std::ifstream ifs;
      ::open_file(prefix + name_ + ".config", ifs);
      ifs >> ne_ >> nd_ >> nh_;
    }

  // Saves parameters to files.
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

  // Registers trainable parameters.
  void register_training(primitiv::Trainer &trainer) {
    trainer.add_parameter(pweh_);
    trainer.add_parameter(pwdh_);
    trainer.add_parameter(pbh_);
    trainer.add_parameter(pwha_);
  }

  // Initializes internal states.
  void init(const std::vector<primitiv::Node> &enc_states) {
    namespace F = primitiv::operators;
    using primitiv::Node;

    const auto weh = F::input<Node>(pweh_);
    e_mat_ = F::concat(enc_states, 1);  // ne_ x len
    eh_mat_ = F::matmul(weh, e_mat_);  // nh_ x len

    wdh_ = F::input<Node>(pwdh_);
    bh_ = F::input<Node>(pbh_);
    wha_ = F::input<Node>(pwha_);
  }

  // Calculates attention probabilities.
  primitiv::Node get_probs(const primitiv::Node &dec_state) {
    namespace F = primitiv::operators;

    const auto dh = F::matmul(wdh_, dec_state) + bh_;  // nh_ x 1
    const auto dh_bc = F::broadcast(dh, 1, eh_mat_.shape()[1]);  // nh_ x len
    const auto h = F::tanh(eh_mat_ + dh_bc);
    const auto a = F::transpose(F::matmul(wha_, h));  // len x 1
    return F::softmax(a, 0);
  }

  // Calculates a context vector.
  primitiv::Node get_context(const primitiv::Node &att_probs) {
    return primitiv::operators::matmul(e_mat_, att_probs);
  }
};

#endif  // MYMT_ATTENTION_H_
