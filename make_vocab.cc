#include <cstdlib>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "messages.pb.h"
#include "utils.h"

using namespace std;

void make_vocab(
    const unsigned vocab_size,
    const string &corpus_file,
    const string &vocab_file) {
  if (vocab_size < 3) {
    throw std::runtime_error("Vocabulary size should be >= 3.");
  }

  cout << "Making vocab from " << corpus_file << " ..." << endl;
  cout << "  size: " << vocab_size << endl;

  ifstream ifs;
  ::open_file(corpus_file, ifs);

  // Counting
  unordered_map<string, unsigned> freq;
  string line;
  unsigned num_all = 0, num_unk = 0;
  while (getline(ifs, line)) {
    for (const auto &w : ::split(line)) {
      ++num_all;
      if (w == "<bos>") throw runtime_error("Corpus has '<bos>' word.");
      else if (w == "<eos>") throw runtime_error("Corpus has '<eos>' word.");
      else if (w == "<unk>") ++num_unk;
      else ++freq[w];
    }
  }

  cout << "  #words: " << num_all << endl;
  cout << "  #explicit <unk>: " << num_unk << endl;

  // Sorting
  using freq_t = pair<string, unsigned>;
  auto cmp = [](const freq_t &a, const freq_t &b) {
    return a.second < b.second;
  };
  priority_queue<freq_t, vector<freq_t>, decltype(cmp)> q(cmp);
  for (const auto &x : freq) q.push(x);

  // Makes vocabulary.
  mymt::messages::Vocabulary vocab;
  mymt::messages::TokenStats *unk_stat = vocab.add_tokens();
  mymt::messages::TokenStats *bos_stat = vocab.add_tokens();
  mymt::messages::TokenStats *eos_stat = vocab.add_tokens();
  unk_stat->set_surface("<unk>");
  bos_stat->set_surface("<bos>");
  eos_stat->set_surface("<eos>");
  bos_stat->set_frequency(0);
  eos_stat->set_frequency(0);

  // Chooses top vocab_size-3 frequent words.
  for (unsigned i = 3; !q.empty() && i < vocab_size; ++i) {
    mymt::messages::TokenStats *stat = vocab.add_tokens();
    stat->set_surface(q.top().first);
    stat->set_frequency(q.top().second);
    num_all -= q.top().second;
    q.pop();
  }

  // Sets <unk>'s frequency.
  cout << "  #actual <unk>: " << num_all << endl;
  unk_stat->set_frequency(num_all);

  ::save_proto(vocab_file, vocab);
  cout << "Saved vocabulary to: " << vocab_file << endl;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    cerr << "Usage: " << argv[0] << endl
         << "    [1] (int) Vocabulay size" << endl
         << "    [2] (file/in) Corpus file" << endl
         << "    [3] (file/out) Vocabulary file" << endl;
    exit(1);
  }

  ::global_try_block([&]() {
      const unsigned vocab_size = stoi(argv[1]);
      const string corpus_file = argv[2];
      const string vocab_file = argv[3];
      ::make_vocab(vocab_size, corpus_file, vocab_file);
  });

  return 0;
}
