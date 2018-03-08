#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <primitiv_nmt/primitiv_nmt.pb.h>
#include <primitiv_nmt/utils.h>
#include <primitiv_nmt/vocabulary.h>

using namespace std;

void dump_corpus(
    const primitiv_nmt::proto::Corpus &corpus,
    const ::Vocabulary &src_vocab, const ::Vocabulary &trg_vocab) {
  for (unsigned i = 0; i < static_cast<unsigned>(corpus.samples_size()); ++i) {
    const auto &sample = corpus.samples()[i];
    cout << "sentence " << i << ':' << endl;
    cout << "  source:" << endl;
    vector<unsigned> src_ids(
        sample.source().token_ids().begin(),
        sample.source().token_ids().end());
    cout << "    id:";
    for (const unsigned id : src_ids) cout << ' ' << id;
    cout << endl;
    cout << "    surface: " << src_vocab.ids_to_line(src_ids) << endl;
    cout << "  target:" << endl;
    vector<unsigned> trg_ids(
        sample.target().token_ids().begin(),
        sample.target().token_ids().end());
    cout << "    id:";
    for (const unsigned id : trg_ids) cout << ' ' << id;
    cout << endl;
    cout << "    surface: " << trg_vocab.ids_to_line(trg_ids) << endl;
  }
}

int main(int argc, char *argv[]) {
  ::check_args(argc, argv, {
      "(file/in) Corpus file",
      "(file/in) Source vocabulary file",
      "(file/in) Target vocabulary file",
  });

  ::global_try_block([&]() {
      primitiv_nmt::proto::Corpus corpus;
      ::load_proto(argv[1], corpus);
      ::Vocabulary src_vocab(argv[2]);
      ::Vocabulary trg_vocab(argv[3]);
      dump_corpus(corpus, src_vocab, trg_vocab);
  });

  return 0;
}
