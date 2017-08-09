#ifndef MYMT_SAMPLER_H_
#define MYMT_SAMPLER_H_

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

#include "messages.pb.h"

struct Batch {
  std::vector<std::vector<unsigned>> source;
  std::vector<std::vector<unsigned>> target;
};

class Sampler {
public:
  void reset() = 0;
  bool has_next() const = 0;
  Batch next() = 0;
};

class MonotoneSampler : public Sampler {
  const mymt::messages::Corpus &corpus_;
  unsigned pos_;

public:
  MonotoneSampler(mymt::messages::Corpus &corpus) : corpus_(corpus), pos_(0) {}

  void reset() override { pos_ = 0; }
  bool has_next() const override { return pos_ < corpus_.samples_size(); }

  Batch next() override {
    if (!has_next()) throw std::runtime_error("No next batch.");
    const auto &sample = corpus_.samples()[pos_];
    Batch batch;
    batch.source.reserve(sample.source().tokens_size());
    for (const auto &token : sample.source().tokens()) {
      batch.source.emplace_back(std::vector<unsigned> {token.id()});
    }
    batch.target.reserve(sample.target().tokens_size());
    for (const auto &token : sample.target().tokens()) {
      batch.target.emplace_back(std::vector<unsigned> {token.id()});
    }
    ++pos;
    return batch;
  }
};

#endif  // MYMT_SAMPLER_H_
