#ifndef MYMT_LSTM_H_
#define MYMT_LSTM_H_

#include <fstream>
#include <string>

#include <primitiv/primitiv.h>

#include "utils.h"

// Hand-written LSTM with input/forget/output gates and no peepholes.
// Formulation:
//   i = sigmoid(W_xi . x[t] + W_hi . h[t-1] + b_i)
//   f = sigmoid(W_xf . x[t] + W_hf . h[t-1] + b_f)
//   o = sigmoid(W_xo . x[t] + W_ho . h[t-1] + b_o)
//   j = tanh   (W_xj . x[t] + W_hj . h[t-1] + b_j)
//   c[t] = i * j + f * c[t-1]
//   h[t] = o * tanh(c[t])
class LSTM {
  std::string name_;
  unsigned ni_, no_;
  float dr_;
  primitiv::Parameter pwxh_, pwhh_, pbh_;
  primitiv::Node wxh_, whh_, bh_, h_, c_;

public:
  LSTM(
      const std::string &name,
      unsigned input_size, unsigned output_size,
      float dropout_rate)
    : name_(name)
    , ni_(input_size)
    , no_(output_size)
    , dr_(dropout_rate)
    , pwxh_(name_ + ".w_xh", {4 * no_, ni_},
        primitiv::initializers::XavierUniform())
    , pwhh_(name_ + ".w_hh", {4 * no_, no_},
        primitiv::initializers::XavierUniform())
    , pbh_(name_ + ".b_h", {4 * no_},
        primitiv::initializers::Constant(0)) {}

  // Loads all parameters.
  LSTM(const std::string &name, const std::string &prefix)
    : name_(name)
    , pwxh_(primitiv::Parameter::load(prefix + name_ + ".w_xh"))
    , pwhh_(primitiv::Parameter::load(prefix + name_ + ".w_hh"))
    , pbh_(primitiv::Parameter::load(prefix + name_ + ".b_h")) {
      std::ifstream ifs;
      ::open_file(prefix + name_ + ".config", ifs);
      ifs >> ni_ >> no_ >> dr_;
  }

  // Saves all parameters.
  void save(const std::string &prefix) const {
    pwxh_.save(prefix + pwxh_.name());
    pwhh_.save(prefix + pwhh_.name());
    pbh_.save(prefix + pbh_.name());
    std::ofstream ofs;
    ::open_file(prefix + name_ + ".config", ofs);
    ofs << ni_ << std::endl;
    ofs << no_ << std::endl;
    ofs << dr_ << std::endl;
  }

  // Adds parameters to the trainer.
  void register_training(primitiv::Trainer &trainer) {
    trainer.add_parameter(pwxh_);
    trainer.add_parameter(pwhh_);
    trainer.add_parameter(pbh_);
  }

  // Initializes internal values.
  void init(
      const primitiv::Node &init_c,
      const primitiv::Node &init_h,
      bool train) {
    namespace F = primitiv::node_ops;
    wxh_ = F::input(pwxh_);
    whh_ = F::dropout(F::input(pwhh_), dr_, train);
    bh_ = F::input(pbh_);
    c_ = init_c.valid() ? init_c : F::zeros({no_});
    h_ = init_h.valid() ? init_h : F::tanh(c_);
  }

  // One step forwarding.
  primitiv::Node forward(const primitiv::Node &x, bool train) {
    namespace F = primitiv::node_ops;
    const auto xx = F::dropout(x, dr_, train);
    const auto u = F::matmul(wxh_, xx) + F::matmul(whh_, h_) + bh_;
    const auto i = F::sigmoid(F::slice(u, 0, 0, no_));
    const auto f = F::sigmoid(1 + F::slice(u, 0, no_, 2 * no_));
    const auto o = F::sigmoid(F::slice(u, 0, 2 * no_, 3 * no_));
    const auto j = F::tanh(F::slice(u, 0, 3 * no_, 4 * no_));
    c_ = i * j + f * c_;
    h_ = o * F::tanh(c_);
    return F::dropout(h_, dr_, train);
  }

  // Retrieves current states.
  primitiv::Node get_c() const { return c_; }
  primitiv::Node get_h() const { return h_; }

  // Retrieves hyperparameters.
  std::string name() const { return name_; }
  unsigned input_size() const { return ni_; }
  unsigned output_size() const { return no_; }
};

#endif  // MYMT_LSTM_H_
