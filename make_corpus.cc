#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "messages.pb.h"
#include "utils.h"
#include "vocabulary.h"

using namespace std;

void make_corpus(
    unsigned min_words, unsigned max_words,
    const string &src_corpus_path, const string &trg_corpus_path,
    const ::Vocabulary &src_vocab, const ::Vocabulary &trg_vocab,
    const string &out_path) {
  ifstream src_ifs, trg_ifs;
  ::open_file(src_corpus_path, src_ifs);
  ::open_file(trg_corpus_path, trg_ifs);

  mymt::messages::Corpus corpus;

  string src_line, trg_line;
  unsigned stored = 0, ignored = 0;
  while(getline(src_ifs, src_line) && getline(trg_ifs, trg_line)) {
    const vector<unsigned> src_ids = src_vocab.line_to_ids(
        "<bos> " + src_line + " <eos>");
    const vector<unsigned> trg_ids = trg_vocab.line_to_ids(
        "<bos> " + trg_line + " <eos>");
    const unsigned src_size = src_ids.size() - 2;
    const unsigned trg_size = trg_ids.size() - 2;
    if (src_size < min_words || src_size > max_words ||
        trg_size < min_words || trg_size > max_words) {
      ++ignored;
      continue;
    }

    mymt::messages::Sample *sample = corpus.add_samples();
    mymt::messages::Sentence *source = sample->mutable_source();
    for (const unsigned src_id : src_ids) {
      mymt::messages::Token *token = source->add_tokens();
      token->set_id(src_id);
    }
    mymt::messages::Sentence *target = sample->mutable_target();
    for (const unsigned trg_id : trg_ids) {
      mymt::messages::Token *token = target->add_tokens();
      token->set_id(trg_id);
    }
    ++stored;
  }

  cout << "Corpus analyzed:" << endl;
  cout << "  #stored sentences: " << stored << endl;
  cout << "  #ignored sentences: " << ignored << endl;

  ::save_proto(out_path, corpus);
  cout << "Corpus saved to: " << out_path << endl;
}

int main(int argc, char *argv[]) {
  vector<string> arg_desc {
    "(int) Minimum #words/sentence",
    "(int) Maximum #words/sentence",
    "(file/in) Source corpus",
    "(file/in) Target corpus",
    "(file/in) Source vocabulary",
    "(file/in) Target vocabulary",
    "(file/out) Corpus file",
  };
  if (argc != arg_desc.size() + 1) {
    cerr << "Usage: " << argv[0] << endl;
    for (unsigned i = 0; i < arg_desc.size(); ++i) {
      cerr << "    [" << (i + 1) << "] " << arg_desc[i] << endl;
    }
    exit(1);
  }

  ::global_try_block([&]() {
      const unsigned min_words = std::stoi(*++argv);
      const unsigned max_words = std::stoi(*++argv);
      const string src_corpus_path = *++argv;
      const string trg_corpus_path = *++argv;
      const ::Vocabulary src_vocab(*++argv);
      const ::Vocabulary trg_vocab(*++argv);
      const string out_path = *++argv;
      ::make_corpus(
          min_words, max_words, src_corpus_path, trg_corpus_path,
          src_vocab, trg_vocab, out_path);
  });

  return 0;
}
