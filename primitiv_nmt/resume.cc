#include "config.h"

#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <primitiv/primitiv.h>
#ifdef PRIMITIV_NMT_USE_CUDA
#include <primitiv/primitiv_cuda.h>
#endif

#include "encoder_decoder.h"
#include "nmt_utils.h"
#include "lstm.h"
#include "primitiv_nmt.pb.h"
#include "sampler.h"
#include "utils.h"
#include "vocabulary.h"

int main(int argc, char *argv[]) {
  ::check_args(argc, argv, {
      "(file/in) Train corpus file",
      "(file/in) Dev corpus file",
      "(file/in) Source vocabulary file",
      "(file/in) Target vocabulary file",
      "(dir/out) Model directory",
      "(int) Last epoch",
      "(int) Number of epochs",
#ifdef PRIMITIV_NMT_USE_CUDA
      "(int) GPU ID",
#endif
  });

  ::global_try_block([&]() {
      const std::string train_corpus_file = *++argv;
      const std::string dev_corpus_file = *++argv;
      const std::string src_vocab_file = *++argv;
      const std::string trg_vocab_file = *++argv;
      const std::string model_dir = *++argv;
      const unsigned last_epoch = std::stoi(*++argv);
      const unsigned num_epochs = std::stoi(*++argv);
#ifdef PRIMITIV_NMT_USE_CUDA
      const unsigned gpu_id = std::stoi(*++argv);
#endif

      const unsigned batch_size = ::load_value<unsigned>(
          model_dir + "/batch_size");

      std::cout << "Loading vocabularies ... " << std::flush;
      const ::Vocabulary src_vocab(src_vocab_file);
      const ::Vocabulary trg_vocab(trg_vocab_file);
      std::cout << "done." << std::endl;

      std::cout << "Loading corpus ... " << std::flush;
      primitiv_nmt::proto::Corpus train_corpus, dev_corpus;
      ::load_proto(train_corpus_file, train_corpus);
      ::load_proto(dev_corpus_file, dev_corpus);
      std::random_device rd;
      ::RandomBatchSampler train_sampler(train_corpus, batch_size, rd());
      ::MonotoneSampler dev_sampler(dev_corpus);
      std::cout << "done." << std::endl;

      std::cout << "Initializing devices ... " << std::flush;
#ifdef PRIMITIV_NMT_USE_CUDA
      primitiv::devices::CUDA dev(gpu_id);
#else
      primitiv::devices::Eigen dev;
#endif
      primitiv::Device::set_default(dev);
      std::cout << "done." << std::endl;

      const std::string last_dir = ::get_model_dir(model_dir, last_epoch);

      std::cout << "Loading model ... " << std::flush;
      ::EncoderDecoder<primitiv::Node> model;
      model.load(last_dir + "/model");
      std::cout << "done." << std::endl;

      std::cout << "Loading trainer ... " << std::flush;
      primitiv::optimizers::Adam opt;
      opt.load(last_dir + "/trainer");
      opt.add(model);
      std::cout << "done." << std::endl;

      NMTTrainer trainer(
          model_dir, src_vocab, trg_vocab, model,
          opt, train_sampler, dev_sampler, last_epoch);

      std::cout << "Restart training." << std::endl;
      for (unsigned i = 0; i < num_epochs; ++i) trainer.train();
      std::cout << "Finished." << std::endl;
  });

  return 0;
}
