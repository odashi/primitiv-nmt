#include <iostream>
#include <stdexcept>
#include <string>

#include "messages.pb.h"
#include "utils.h"

using namespace std;

void dump_vocab(const string &path) {
  mymt::proto::Vocabulary vocab;
  ::load_proto(path, vocab);

  unsigned id = 0;
  for (const mymt::proto::TokenStats &token : vocab.tokens()) {
    cout << id++ << '\t' << token.surface() << '\t'
         << token.frequency() << endl;
  }
}

int main(int argc, char *argv[]) {
  ::check_args(argc, argv, {
      "(file/in) Vocabulary file",
  });

  ::global_try_block([&]() {
      ::dump_vocab(argv[1]);
  });

  return 0;
}
