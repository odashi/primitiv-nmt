#ifndef MYMT_UTILS_H_
#define MYMT_UTILS_H_

#include <cctype>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>

template <class FStreamT>
inline void open_file(const std::string &path, FStreamT &fs) {
  fs.open(path);
  if (!fs.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
}

void make_directory(const std::string &path) {
  if (::mkdir(path.c_str(), 0755) != 0) {
    std::string errstr = std::strerror(errno);
    throw std::runtime_error("Failed to make directory: " + path + ": " + errstr);
  }
}

inline std::string trim(std::string &str) {
  unsigned l = 0;
  while (l < str.size() && std::isspace(str[l])) ++l;
  unsigned r = str.size();
  while (r > l && std::isspace(str[r - 1])) --r;
  return str.substr(l, r - l);
}

inline std::vector<std::string> split(const std::string &str) {
  std::vector<std::string> ret;
  unsigned l = 0;
  while (true) {
    while (l < str.size() && std::isspace(str[l])) ++l;
    if (l == str.size()) break;
    unsigned r = l + 1;
    while (r < str.size() && !std::isspace(str[r])) ++r;
    ret.emplace_back(str.substr(l, r - l));
    l = r;
  }
  return ret;
}

inline unsigned argmax(const std::vector<float> &scores) {
  if (scores.empty()) throw std::runtime_error("No scores to calculate argmax.");
  unsigned max_id = 0;
  float max_score = scores[0];
  for (unsigned i = 1; i < scores.size(); ++i) {
    if (scores[i] > max_score) {
      max_id = i;
      max_score = scores[i];
    }
  }
  return max_id;
}

#endif  // MYMT_UTILS_H_
