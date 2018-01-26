#include "config.h"

#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <primitiv/primitiv.h>

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
      "(int) Embedding size",
      "(int) Hidden size",
      "(int) Batch size",
      "(float) Learning rate",
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
      const unsigned embed_size = std::stoi(*++argv);
      const unsigned hidden_size = std::stoi(*++argv);
      const unsigned batch_size = std::stoi(*++argv);
      const float learning_rate = std::stof(*++argv);
      const unsigned num_epochs = std::stoi(*++argv);
#ifdef PRIMITIV_NMT_USE_CUDA
      const unsigned gpu_id = std::stoi(*++argv);
#endif

      ::make_directory(model_dir);
      ::save_value(model_dir + "/batch_size", batch_size);
      ::save_value(model_dir + "/best.epoch", 0);
      ::save_value(model_dir + "/best.dev_avg_loss", 1e10f);

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

      std::cout << "Initializing model ... " << std::flush;
      ::EncoderDecoder<primitiv::Node> model;
      model.init(src_vocab.size(), trg_vocab.size(), embed_size, hidden_size);
      std::cout << "done." << std::endl;

      std::cout << "Initializing trainer ... " << std::flush;
      primitiv::optimizers::Adam opt(learning_rate);
      opt.set_weight_decay(1e-6);
      opt.set_gradient_clipping(5);
      opt.add(model);
      std::cout << "done." << std::endl;

      NMTTrainer trainer(
          model_dir, src_vocab, trg_vocab, model,
          opt, train_sampler, dev_sampler, 0);

      std::cout << "Saving initial model ... " << std::flush;
      trainer.save(
          1e10, 1e10, std::vector<std::string>(dev_corpus.samples_size()));
      std::cout << "done." << std::endl;

      std::cout << "Start training." << std::endl;
      for (unsigned i = 0; i < num_epochs; ++i) trainer.train();
      std::cout << "Finished." << std::endl;
  });

  return 0;
}
