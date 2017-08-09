#ifndef PRIMITIV_NMT_VOCAB_H_
#define PRIMITIV_NMT_VOCAB_H_

#include <fstream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "primitiv_nmt.pb.h"
#include "utils.h"

class Vocabulary {
  std::unordered_map<std::string, unsigned> stoi_;
  std::vector<std::string> itos_;
  std::vector<unsigned> freq_;
  std::vector<float> prob_;

  Vocabulary(const Vocabulary &) = delete;
  Vocabulary &operator=(const Vocabulary &) = delete;

public:
  Vocabulary(const std::string &path) {
    primitiv_nmt::proto::Vocabulary vocab_data;
    ::load_proto(path, vocab_data);
    unsigned sum_freq = 0;
    for (const auto &stat : vocab_data.tokens()) {
      stoi_.emplace(stat.surface(), stoi_.size());
      itos_.emplace_back(stat.surface());
      freq_.emplace_back(stat.frequency());
      sum_freq += stat.frequency();
    }
    for (unsigned f : freq_) {
      prob_.emplace_back(f / static_cast<float>(sum_freq));
    }
  }

  unsigned stoi(const std::string &word) const {
    const auto it = stoi_.find(word);
    return it == stoi_.end() ? 0 : it->second;
  }

  std::string itos(unsigned id) const {
    if (id > size()) {
      throw std::runtime_error("Index out of range: " + std::to_string(id));
    }
    return itos_[id];
  }

  std::vector<unsigned> line_to_ids(const std::string &line) const {
    const std::vector<std::string> words = ::split(line);
    std::vector<unsigned> ids;
    ids.reserve(words.size());
    for (const std::string &word : words) ids.emplace_back(stoi(word));
    return ids;
  }

  std::string ids_to_line(const std::vector<unsigned> &ids) const {
    if (ids.empty()) return std::string();
    std::string ret = itos(ids[0]);
    for (unsigned i = 1; i < ids.size(); ++i) ret += ' ' + itos(ids[i]);
    return ret;
  }

  unsigned freq(unsigned id) const {
    if (id > size()) {
      throw std::runtime_error("Index out of range: " + std::to_string(id));
    }
    return freq_[id];
  }

  float prob(unsigned id) const {
    if (id > size()) {
      throw std::runtime_error("Index out of range: " + std::to_string(id));
    }
    return prob_[id];
  }

  unsigned size() const { return itos_.size(); }
};

#endif  // PRIMITIV_NMT_VOCAB_H_
