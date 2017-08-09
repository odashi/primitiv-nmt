#ifndef PRIMITIV_NMT_ATTENTION_H_
#define PRIMITIV_NMT_ATTENTION_H_

#include <fstream>
#include <string>
#include <vector>

#include <primitiv/primitiv.h>

#include "utils.h"

// Multilayer perceptron-based attention
template<typename Var>
class Attention : public primitiv::Model {
  primitiv::Parameter pweh_, pwdh_, pbh_, pwha_;
  Var e_mat_, eh_mat_, wdh_, bh_, wha_;

public:
  // New object.
  Attention() {
    add("weh", pweh_);
    add("wdh", pwdh_);
    add("bh", pbh_);
    add("wha", pwha_);
  }

  // Initializes parameters.
  void init(
      unsigned encoder_size, unsigned decoder_size, unsigned hidden_size) {
    namespace I = primitiv::initializers;
    pweh_.init({hidden_size, encoder_size}, I::Uniform(-0.1, 0.1));
    pwdh_.init({hidden_size, decoder_size}, I::Uniform(-0.1, 0.1));
    pbh_.init({hidden_size}, I::Constant(0));
    pwha_.init({1, hidden_size}, I::Uniform(-0.1, 0.1));
  }

  // Initializes internal states.
  void reset(const std::vector<Var> &enc_states) {
    namespace F = primitiv::functions;
    const Var weh = F::parameter<Var>(pweh_);
    e_mat_ = F::concat(enc_states, 1);  // {enc_size, len}
    eh_mat_ = F::matmul(weh, e_mat_);  // {h_size, len}
    wdh_ = F::parameter<Var>(pwdh_);
    bh_ = F::parameter<Var>(pbh_);
    wha_ = F::parameter<Var>(pwha_);
  }

  // Calculates attention probabilities.
  Var get_probs(const Var &dec_state) {
    namespace F = primitiv::functions;
    const Var dh = F::matmul(wdh_, dec_state) + bh_;  // {h_size}
    const Var dh_bc = F::broadcast(dh, 1, eh_mat_.shape()[1]);  // {h_size, len}
    const Var h = F::tanh(eh_mat_ + dh_bc);
    const Var a = F::transpose(F::matmul(wha_, h));  // {len}
    return F::softmax(a, 0);
  }

  // Calculates a context vector.
  Var get_context(const Var &att_probs) {
    return primitiv::functions::matmul(e_mat_, att_probs);
  }

  // Retrieves hyperparameters
  unsigned encoder_size() const { return pweh_.shape()[1]; }
  unsigned decoder_size() const { return pwdh_.shape()[1]; }
  unsigned hidden_size() const { return pweh_.shape()[0]; }
};

#endif  // PRIMITIV_NMT_ATTENTION_H_
