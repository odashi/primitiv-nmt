#ifndef MYMT_VOCAB_H_
#define MYMT_VOCAB_H_

#include <fstream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils.h"

class Vocab {
  std::unordered_map<std::string, unsigned> stoi_;
  std::vector<std::string> itos_;
  std::vector<unsigned> freq_;

public:
  static Vocab make(const std::string &path, unsigned size) {
    if (size < 3) {
      throw std::runtime_error(
          "Vocab size should be equal-to or greater-than 3.");
    }

    std::cout << "Loading vocab from " << path << " ..." << std::endl;
    std::cout << "  size: " << size << std::endl;

    std::ifstream ifs;
    ::open_file(path, ifs);

    // Counting
    std::unordered_map<std::string, unsigned> freq;
    std::string line;
    unsigned num_all = 0, num_unk = 0;
    while (std::getline(ifs, line)) {
      for (const auto &w : ::split(line)) {
        ++num_all;
        if (w == "<bos>") throw std::runtime_error("Corpus has '<bos>' word.");
        else if (w == "<eos>") throw std::runtime_error("Corpus has '<eos>' word.");
        else if (w == "<unk>") ++num_unk;
        else ++freq[w];
      }
    }

    std::cout << "  #words: " << num_all << std::endl;
    std::cout << "  #explicit <unk>: " << num_unk << std::endl;

    // Sorting
    using freq_t = std::pair<std::string, unsigned>;
    auto cmp = [](const freq_t &a, const freq_t &b) { return a.second < b.second; };
    std::priority_queue<freq_t, std::vector<freq_t>, decltype(cmp)> q(cmp);
    for (const auto &x : freq) q.push(x);

    Vocab vocab;
    vocab.stoi_.insert(std::make_pair("<unk>", 0));
    vocab.stoi_.insert(std::make_pair("<bos>", 1));
    vocab.stoi_.insert(std::make_pair("<eos>", 2));
    vocab.itos_.emplace_back("<unk>");
    vocab.itos_.emplace_back("<bos>");
    vocab.itos_.emplace_back("<eos>");
    vocab.freq_.emplace_back(0);
    vocab.freq_.emplace_back(0);
    vocab.freq_.emplace_back(0);

    // Chooses top size-3 frequent words.
    for (unsigned i = 3; !q.empty() && i < size; ++i) {
      vocab.stoi_.insert(make_pair(q.top().first, i));
      vocab.itos_.emplace_back(q.top().first);
      vocab.freq_.emplace_back(q.top().second);
      num_all -= q.top().second;
      q.pop();
    }

    std::cout << "  #actual <unk>: " << num_all << std::endl;

    vocab.freq_[0] = num_all;

    return vocab;
  }

  static Vocab load(const std::string &path) {
    std::ifstream ifs;
    ::open_file(path, ifs);

    Vocab vocab;
    std::string line;
    while (getline(ifs, line)) {
      std::stringstream ss(line);
      std::string word;
      unsigned freq;
      ss >> word >> freq;
      vocab.stoi_.insert(std::make_pair(word, vocab.stoi_.size()));
      vocab.itos_.emplace_back(word);
      vocab.freq_.emplace_back(freq);
    }

    std::cout << "Loaded vocab from: " << path << std::endl;

    return vocab;
  }

  void save(const std::string &path) {
    std::ofstream ofs;
    ::open_file(path, ofs);
    for (unsigned i = 0; i < itos_.size(); ++i) {
      ofs << itos_[i] << '\t' << freq_[i] << std::endl;
    }

    std::cout << "Saved vocab to: " << path << std::endl;
  }

  unsigned stoi(const std::string &word) const {
    const auto it = stoi_.find(word);
    return it == stoi_.end() ? 0 : it->second;
  }

  std::string itos(unsigned id) const {
    if (id > itos_.size()) {
      throw std::runtime_error("Vocab out of range: " + std::to_string(id));
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
    if (id > freq_.size()) {
      throw std::runtime_error("Vocab out of range: " + std::to_string(id));
    }
    return freq_[id];
  }

  unsigned size() const { return itos_.size(); }
};

#endif  // MYMT_VOCAB_H_
