#ifndef PRIMITIV_NMT_AFFINE_H_
#define PRIMITIV_NMT_AFFINE_H_

#include <fstream>
#include <string>

#include <primitiv/primitiv.h>

#include "utils.h"

// Affine transform
template<typename Var>
class Affine : public primitiv::Model {
  primitiv::Parameter pw_, pb_;
  Var w_, b_;

public:
  // New model.
  Affine() {
    add("w", pw_);
    add("b", pb_);
  }

  // Initializes parameters.
  void init(unsigned input_size, unsigned output_size) {
    namespace I = primitiv::initializers;
    pw_.init({output_size, input_size}, I::Uniform(-0.1, 0.1));
    pb_.init({output_size}, I::Constant(0));
  }

  // Initializes internal values.
  void reset() {
    namespace F = primitiv::functions;
    w_ = F::parameter<Var>(pw_);
    b_ = F::parameter<Var>(pb_);
  }

  // Applies transformation.
  Var forward(const Var &x) {
    return primitiv::functions::matmul(w_, x) + b_;
  }

  // Retrieves hyperparameters.
  unsigned input_size() const { return pw_.shape()[1]; }
  unsigned output_size() const { return pw_.shape()[0]; }
};

#endif  // PRIMITIV_NMT_AFFINE_H_
