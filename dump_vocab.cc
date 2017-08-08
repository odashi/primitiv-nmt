#include <iostream>
#include <stdexcept>
#include <string>

#include "messages.pb.h"
#include "utils.h"

using namespace std;

void dump_vocab(const string &path) {
  mymt::messages::Vocabulary vocab;
  ::load_proto(path, vocab);

  unsigned id = 0;
  for (const mymt::messages::TokenStats &token : vocab.tokens()) {
    cout << id++ << '\t' << token.surface() << '\t'
         << token.frequency() << endl;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << endl
         << "    [1] (file/in) Vocabulary file" << endl;
    exit(1);
  }

  ::global_try_block([&]() {
      ::dump_vocab(argv[1]);
  });

  return 0;
}
