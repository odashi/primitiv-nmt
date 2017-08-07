// Encoder-decoder with attention.

#include <cstdio>
#include <iostream>
#include <string>

#include <sys/stat.h>

#include <primitiv/primitiv.h>
#include <primitiv/primitiv_cuda.h>

#include "vocab.h"
#include "utils.h"

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::vector;

void train_main(int argc, char *argv[]) {
  vector<string> options {
    "Model directory",
    "Train source corpus",
    "Train target corpus",
    "Valid source corpus",
    "Valid target corpus",
    "Vocabualry size",
  };

  if (argc != options.size() + 2) {
    cerr << "Usage: " << argv[0] << " train" << endl;
    for (unsigned i = 0; i < options.size(); ++i) {
      cerr << "    [" << (i + 1) << "] " << options[i] << endl;
    }
    exit(1);
  }

  argv += 2;
  const string model_dir = *argv++;
  const string train_src_path = *argv++;
  const string train_trg_path = *argv++;
  const string valid_src_path = *argv++;
  const string valid_trg_path = *argv++;
  const unsigned vocab_size = std::stoi(*argv++);

  if (::mkdir(model_dir.c_str(), 0755) != 0) {
    std::perror("mkdir() failed");
    exit(1);
  }

  Vocab src_vocab = Vocab::make(train_src_path, vocab_size);
  src_vocab.save(model_dir + "/source.vocab");

  Vocab trg_vocab = Vocab::make(train_trg_path, vocab_size);
  trg_vocab.save(model_dir + "/target.vocab");
}

void resume_main(int argc, char *argv[]) {
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " resume" << endl;
    cerr << "    [1] Model directory" << endl;
    exit(1);
  }
}

void test_main(int argc, char *argv[]) {
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " test" << endl;
    cerr << "    [1] Model directory" << endl;
    exit(1);
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " (train|resume|test) [options]" << endl;
    exit(1);
  }

  const string mode = argv[1];

  if (mode == "train") ::train_main(argc, argv);
  else if (mode == "resume") ::resume_main(argc, argv);
  else if (mode == "test") ::test_main(argc, argv);
  else {
    cerr << "unknown mode: " << mode << endl;
    exit(1);
  }

  return 0;
}

