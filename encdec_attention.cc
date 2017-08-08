// Encoder-decoder with attention.

#include <iostream>
#include <stdexcept>
#include <string>

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
    "Source train corpus",
    "Target train corpus",
    "Source validation corpus",
    "Target validation corpus",
    "Source vocabualry size",
    "Target vocabualry size",
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
  const string src_train_path = *argv++;
  const string trg_train_path = *argv++;
  const string src_valid_path = *argv++;
  const string trg_valid_path = *argv++;
  const unsigned src_vocab_size = std::stoi(*argv++);
  const unsigned trg_vocab_size = std::stoi(*argv++);

  ::make_directory(model_dir);

  Vocab src_vocab = Vocab::make(src_train_path, src_vocab_size);
  src_vocab.save(model_dir + "/source.vocab");

  Vocab trg_vocab = Vocab::make(trg_train_path, trg_vocab_size);
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

  try {
    if (mode == "train") ::train_main(argc, argv);
    else if (mode == "resume") ::resume_main(argc, argv);
    else if (mode == "test") ::test_main(argc, argv);
    else throw std::runtime_error("Unknown mode: " + mode);
  } catch (std::exception &ex) {
    cerr << "Caught std::exception." << endl;
    cerr << "  what(): " << ex.what() << endl;
    exit(1);
  } catch (...) {
    cerr << "Caught unknown exception." << endl;
    exit(1);
  }


  return 0;
}

