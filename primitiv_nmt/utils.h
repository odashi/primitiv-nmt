#ifndef PRIMITIV_NMT_UTILS_H_
#define PRIMITIV_NMT_UTILS_H_

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>

inline void check_args(
    int argc, char *argv[], const std::vector<std::string> &desc) {
  if (static_cast<unsigned>(argc) != desc.size() + 1) {
    std::cerr << "Usage: " << argv[0] << std::endl;
    for (unsigned i = 0; i < desc.size(); ++i) {
      std::cerr << "    [" << (i + 1) << "] " << desc[i] << std::endl;
    }
    std::exit(1);
  }
}

inline void global_try_block(std::function<void()> subroutine) {
  try {
    subroutine();
  } catch (std::exception &ex) {
    std::cerr << "Caught std::exception." << std::endl;
    std::cerr << "  what(): " << ex.what() << std::endl;
    std::exit(1);
  } catch (...) {
    std::cerr << "Caught unknown exception." << std::endl;
    std::exit(1);
  }
}

template <class FStreamT>
inline void open_file(const std::string &path, FStreamT &fs) {
  fs.open(path);
  if (!fs.is_open()) {
    throw std::runtime_error(
        "Failed to open file: " + path + ": " + std::strerror(errno));
  }
}

inline void make_directory(const std::string &path) {
  if (::mkdir(path.c_str(), 0755) != 0) {
    throw std::runtime_error(
        "Failed to make directory: " + path + ": " + std::strerror(errno));
  }
}

template <typename T>
inline void save_value(const std::string &path, T value) {
  std::ofstream ofs;
  ::open_file(path, ofs);
  ofs << value << std::endl;
}

template <>
inline void save_value(const std::string &path, float value) {
  std::ofstream ofs;
  ::open_file(path, ofs);
  char buf[16];
  ::sprintf(buf, "%.8e", value);
  ofs << buf << std::endl;
}

template <typename T>
inline T load_value(const std::string &path) {
  std::ifstream ifs;
  ::open_file(path, ifs);
  T value;
  ifs >> value;
  return value;
}

inline void save_strings(
    const std::string &path, const std::vector<std::string> &strs) {
  std::ofstream ofs;
  ::open_file(path, ofs);
  for (const std::string &str : strs) ofs << str << std::endl;
}

template <class ProtoT>
inline void save_proto(const std::string &path, const ProtoT &proto) {
  std::ofstream ofs;
  ::open_file(path, ofs);
  if (!proto.SerializeToOstream(&ofs)) {
    throw std::runtime_error("Failed to save proto: " + path);
  }
}

template <class ProtoT>
inline void load_proto(const std::string &path, ProtoT &proto) {
  std::ifstream ifs;
  ::open_file(path, ifs);
  if (!proto.ParseFromIstream(&ifs)) {
    throw std::runtime_error("Failed to load proto: " + path);
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

inline std::vector<std::string> split(
    const std::string &str, const char delim) {
  std::vector<std::string> ret;
  unsigned l = 0;
  while (l <= str.size()) {
    unsigned r = l;
    while (r < str.size() && str[r] != delim) ++r;
    ret.emplace_back(str.substr(l, r - l));
    l = r + 1;
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

inline std::string get_model_dir(const std::string &prefix, unsigned epoch) {
  char buf[8];
  std::sprintf(buf, "%04u", epoch);
  return prefix + '/' + buf;
}

#endif  // PRIMITIV_NMT_UTILS_H_
