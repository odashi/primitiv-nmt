#ifndef MYMT_AFFINE_H_
#define MYMT_AFFINE_H_

#include <fstream>
#include <string>

#include <primitiv/primitiv.h>

#include "utils.h"

// Affine transform
class Affine {
  std::string name_;
  unsigned ni_, no_;
  primitiv::Parameter pw_, pb_;
  primitiv::Node w_, b_;

public:
  // New model.
  Affine(const std::string &name, unsigned input_size, unsigned output_size)
    : name_(name)
    , ni_(input_size)
    , no_(output_size)
    , pw_(name_ + ".w", {no_, ni_}, primitiv::initializers::XavierUniform())
    , pb_(name_ + ".b", {no_}, primitiv::initializers::Constant(0)) {}

  // Loads all parameters.
  Affine(const std::string &name, const std::string &prefix)
    : name_(name)
    , pw_(primitiv::Parameter::load(prefix + name_ + ".w"))
    , pb_(primitiv::Parameter::load(prefix + name_ + ".b")) {
      std::ifstream ifs;
      ::open_file(prefix + name_ + ".config", ifs);
      ifs >> ni_ >> no_;
    }

  // Saves all parameters.
  void save(const std::string &prefix) const {
    pw_.save(prefix + pw_.name());
    pb_.save(prefix + pb_.name());
    std::ofstream ofs;
    ::open_file(prefix + name_ + ".config", ofs);
    ofs << ni_ << std::endl;
    ofs << no_ << std::endl;
  }

  // Adds parameters to the trainer.
  void register_training(primitiv::Trainer &trainer) {
    trainer.add_parameter(pw_);
    trainer.add_parameter(pb_);
  }

  // Initializes internal values.
  void init() {
    namespace F = primitiv::operators;
    using primitiv::Node;
    w_ = F::input<Node>(pw_);
    b_ = F::input<Node>(pb_);
  }

  // Applies transformation.
  primitiv::Node forward(const primitiv::Node &x) {
    return primitiv::operators::matmul(w_, x) + b_;
  }

  // Retrieves hyperparameters.
  std::string name() const { return name_; }
  unsigned input_size() const { return ni_; }
  unsigned output_size() const { return no_; }
};

#endif  // MYMT_AFFINE_H_
