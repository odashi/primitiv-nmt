#ifndef PRIMITIV_NMT_SAMPLER_H_
#define PRIMITIV_NMT_SAMPLER_H_

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

#include "primitiv_nmt.pb.h"

struct Batch {
  std::vector<std::vector<unsigned>> source;
  std::vector<std::vector<unsigned>> target;
};

class Sampler {
  Sampler(const Sampler &) = delete;
  Sampler &operator=(const Sampler &) = delete;

public:
  Sampler() = default;
  virtual void reset() = 0;
  virtual Batch next() = 0;
  virtual bool has_next() const = 0;
  virtual unsigned num_sentences() const = 0;
};

class MonotoneSampler : public Sampler {
  const primitiv_nmt::proto::Corpus &corpus_;
  unsigned pos_;

  MonotoneSampler(const MonotoneSampler &) = delete;
  MonotoneSampler &operator=(const MonotoneSampler &) = delete;

public:
  MonotoneSampler(const primitiv_nmt::proto::Corpus &corpus)
    : corpus_(corpus), pos_(0) {}

  void reset() override { pos_ = 0; }

  Batch next() override {
    if (!has_next()) throw std::runtime_error("No next batch.");
    const auto &sample = corpus_.samples()[pos_];
    Batch batch;
    batch.source.reserve(sample.source().token_ids_size());
    for (const unsigned tok_id : sample.source().token_ids()) {
      batch.source.emplace_back(std::vector<unsigned> {tok_id});
    }
    batch.target.reserve(sample.target().token_ids_size());
    for (const unsigned tok_id : sample.target().token_ids()) {
      batch.target.emplace_back(std::vector<unsigned> {tok_id});
    }
    ++pos_;
    return batch;
  }

  bool has_next() const override {
    return pos_ < static_cast<unsigned>(corpus_.samples_size());
  }

  unsigned num_sentences() const override { return corpus_.samples_size(); }
};

class RandomBatchSampler : public Sampler {
  const primitiv_nmt::proto::Corpus &corpus_;
  unsigned bs_;
  std::mt19937 rng_;
  std::vector<unsigned> ids_;
  std::vector<std::pair<unsigned, unsigned>> ranges_;
  unsigned pos_;

  RandomBatchSampler(const RandomBatchSampler &) = delete;
  RandomBatchSampler &operator=(const RandomBatchSampler &) = delete;

public:
  RandomBatchSampler(
      const primitiv_nmt::proto::Corpus &corpus, unsigned batch_size, unsigned seed)
    : corpus_(corpus)
    , bs_(batch_size)
    , rng_(seed)
    , ids_(corpus.samples_size())
    , pos_(0) {
      std::iota(ids_.begin(), ids_.end(), 0);
    }

  void reset() override {
    ranges_.clear();
    std::shuffle(ids_.begin(), ids_.end(), rng_);
    std::sort(ids_.begin(), ids_.end(), [&](unsigned a, unsigned b) {
        const auto &sa = corpus_.samples()[a];
        const auto &sb = corpus_.samples()[b];
        const unsigned sa_src = sa.source().token_ids_size();
        const unsigned sb_src = sb.source().token_ids_size();
        const unsigned sa_trg = sa.target().token_ids_size();
        const unsigned sb_trg = sb.target().token_ids_size();
        if (sa_src == sb_src) return sa_trg < sb_trg;
        else return sa_src < sb_src;
    });
    const unsigned num_total_samples = corpus_.samples_size();
    unsigned left = 0;
    while (left < num_total_samples) {
      const auto &left_sample = corpus_.samples()[ids_[left]];
      const unsigned left_src = left_sample.source().token_ids_size();
      const unsigned left_trg = left_sample.target().token_ids_size();
      unsigned right = left + 1;
      while (right < num_total_samples) {
        const auto &right_sample = corpus_.samples()[ids_[right]];
        const unsigned right_src = right_sample.source().token_ids_size();
        const unsigned right_trg = right_sample.target().token_ids_size();
        if (right_src != left_src || right_trg != left_trg) break;
        ++right;
      }
      const unsigned num_sents = right - left;
      const unsigned num_batches = (num_sents + bs_ - 1) / bs_;
      const unsigned num_sents_per_batch = num_sents / num_batches;
      const unsigned carry = num_sents % num_batches;
      unsigned first = left;
      for (unsigned i = 0; i < num_batches; ++i) {
        const unsigned second = first + num_sents_per_batch + (i < carry);
        ranges_.emplace_back(first, second);
        first = second;
      }
      left = right;
    }
    std::shuffle(ranges_.begin(), ranges_.end(), rng_);
    pos_ = 0;
  }

  Batch next() override {
    if (!has_next()) throw std::runtime_error("No next batch.");
    const unsigned first = ranges_[pos_].first;
    const unsigned second = ranges_[pos_].second;
    const unsigned batch_size = second - first;
    const auto &first_sample = corpus_.samples()[ids_[first]];
    const unsigned src_len = first_sample.source().token_ids_size();
    const unsigned trg_len = first_sample.target().token_ids_size();
    Batch batch {
      std::vector<std::vector<unsigned>>(
          src_len, std::vector<unsigned>(batch_size)),
      std::vector<std::vector<unsigned>>(
          trg_len, std::vector<unsigned>(batch_size)),
    };
    for (unsigned i = 0; i < batch_size; ++i) {
      const auto &sample = corpus_.samples()[ids_[i + first]];
      const auto &src_token_ids = sample.source().token_ids();
      for (unsigned j = 0; j < src_len; ++j) {
        batch.source[j][i] = src_token_ids[j];
      }
      const auto &trg_token_ids = sample.target().token_ids();
      for (unsigned j = 0; j < trg_len; ++j) {
        batch.target[j][i] = trg_token_ids[j];
      }
    }
    ++pos_;
    return batch;
  }

  bool has_next() const override { return pos_ < ranges_.size(); }

  unsigned num_sentences() const override { return corpus_.samples_size(); }
};

#endif  // PRIMITIV_NMT_SAMPLER_H_
