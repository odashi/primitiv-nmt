#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "messages.pb.h"
#include "utils.h"
#include "vocabulary.h"

using namespace std;

void dump_corpus(
    const mymt::messages::Corpus &corpus,
    const ::Vocabulary &src_vocab, const ::Vocabulary &trg_vocab) {
  for (unsigned i = 0; i < corpus.samples_size(); ++i) {
    const auto &sample = corpus.samples()[i];
    cout << "sentence " << i << ':' << endl;
    cout << "  source:" << endl;
    vector<unsigned> src_ids;
    cout << "    id:";
    for (const auto &tok : sample.source().tokens()) {
      cout << ' ' << tok.id();
      src_ids.emplace_back(tok.id());
    }
    cout << endl;
    cout << "    surface: " << src_vocab.ids_to_line(src_ids) << endl;
    cout << "  target:" << endl;
    vector<unsigned> trg_ids;
    cout << "    id:";
    for (const auto &tok : sample.target().tokens()) {
      cout << ' ' << tok.id();
      trg_ids.emplace_back(tok.id());
    }
    cout << endl;
    cout << "    surface: " << trg_vocab.ids_to_line(trg_ids) << endl;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    cerr << "Usage: " << argv[0] << endl
         << "    [1] (file/in) Corpus file" << endl
         << "    [2] (file/in) Source vocabulary" << endl
         << "    [3] (file/in) Target vocabulary" << endl;
    exit(1);
  }

  ::global_try_block([&]() {
      mymt::messages::Corpus corpus;
      ::load_proto(argv[1], corpus);
      ::Vocabulary src_vocab(argv[2]);
      ::Vocabulary trg_vocab(argv[3]);
      dump_corpus(corpus, src_vocab, trg_vocab);
  });

  return 0;
}