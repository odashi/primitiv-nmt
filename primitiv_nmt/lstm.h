#ifndef PRIMITIV_NMT_LSTM_H_
#define PRIMITIV_NMT_LSTM_H_

#include <cmath>
#include <fstream>
#include <string>

#include <primitiv/primitiv.h>

#include <primitiv_nmt/utils.h>

// Hand-written LSTM with input/forget/output gates and no peepholes.
// Formulation:
//   i = sigmoid(W_xi . x[t] + W_hi . h[t-1] + b_i)
//   f = sigmoid(W_xf . x[t] + W_hf . h[t-1] + b_f)
//   o = sigmoid(W_xo . x[t] + W_ho . h[t-1] + b_o)
//   j = tanh   (W_xj . x[t] + W_hj . h[t-1] + b_j)
//   c[t] = i * j + f * c[t-1]
//   h[t] = o * tanh(c[t])
template<typename Var>
class LSTM : public primitiv::Model {
  primitiv::Parameter pwxh_, pwhh_, pbh_;
  Var wxh_, whh_, bh_, h_, c_;

public:
  // New model.
  LSTM() {
    add("wxh", pwxh_);
    add("whh", pwhh_);
    add("bh", pbh_);
  }

  // Initializes parameters.
  void init(unsigned input_size, unsigned output_size) {
    namespace I = primitiv::initializers;
    pwxh_.init({4 * output_size, input_size}, I::Uniform(-0.1, 0.1));
    pwhh_.init({4 * output_size, output_size}, I::Uniform(-0.1, 0.1));
    pbh_.init({4 * output_size}, I::Constant(0));
  }

  // Initializes internal values.
  void reset(const Var &init_c, const Var &init_h) {
    namespace F = primitiv::functions;
    wxh_ = F::parameter<Var>(pwxh_);
    whh_ = F::parameter<Var>(pwhh_);
    bh_ = F::parameter<Var>(pbh_);
    c_ = init_c.valid() ? init_c : F::zeros<Var>({output_size()});
    h_ = init_h.valid() ? init_h : F::tanh(c_);
  }

  // One step forwarding.
  Var forward(const Var &x) {
    namespace F = primitiv::functions;
    const unsigned no = output_size();
    const Var u = F::matmul(wxh_, x) + F::matmul(whh_, h_) + bh_;
    const Var i = F::sigmoid(F::slice(u, 0, 0, no));
    const Var f = F::sigmoid(1 + F::slice(u, 0, no, 2 * no));
    const Var o = F::sigmoid(F::slice(u, 0, 2 * no, 3 * no));
    const Var j = F::tanh(F::slice(u, 0, 3 * no, 4 * no));
    c_ = i * j + f * c_;
    h_ = o * F::tanh(c_);
    return h_;
  }

  // Retrieves current states.
  Var get_c() const { return c_; }
  Var get_h() const { return h_; }

  // Retrieves hyperparameters.
  unsigned input_size() const { return pwxh_.shape()[1]; }
  unsigned output_size() const { return pwxh_.shape()[0] / 4; }
};

#endif  // PRIMITIV_NMT_LSTM_H_
