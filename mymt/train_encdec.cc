#include "config.h"

#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <primitiv/primitiv.h>
#ifdef MYMT_USE_CUDA
#include <primitiv/primitiv_cuda.h>
#endif

#include "attention_encoder_decoder.h"
#include "nmt_utils.h"
#include "lstm.h"
#include "mymt.pb.h"
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
      "(float) dropout_rate",
      "(int) Batch size",
      "(int) Number of epochs",
  });

  ::global_try_block([&]() {
      const std::string train_corpus_file = *++argv;
      const std::string dev_corpus_file = *++argv;
      const std::string src_vocab_file = *++argv;
      const std::string trg_vocab_file = *++argv;
      const std::string model_dir = *++argv;
      const unsigned embed_size = std::stoi(*++argv);
      const unsigned hidden_size = std::stoi(*++argv);
      const float dropout_rate = std::stof(*++argv);
      const unsigned batch_size = std::stoi(*++argv);
      const unsigned num_epochs = std::stoi(*++argv);

      std::cout << "Loading vocabularies ... " << std::flush;
      const ::Vocabulary src_vocab(src_vocab_file);
      const ::Vocabulary trg_vocab(trg_vocab_file);
      std::cout << "done." << std::endl;

      std::cout << "Loading corpus ... " << std::flush;
      mymt::proto::Corpus train_corpus, dev_corpus;
      ::load_proto(train_corpus_file, train_corpus);
      ::load_proto(dev_corpus_file, dev_corpus);
      std::random_device rd;
      ::RandomBatchSampler train_sampler(train_corpus, batch_size, rd());
      ::MonotoneSampler dev_sampler(dev_corpus);
      std::cout << "done." << std::endl;

      std::cout << "Initializing devices ... " << std::flush;
#ifdef MYMT_USE_CUDA
      primitiv::CUDADevice dev(0);
#else
      primitiv::CPUDevice dev;
#endif
      primitiv::Device::set_default_device(dev);
      std::cout << "done." << std::endl;

      std::cout << "Initializing model ... " << std::flush;
      ::AttentionEncoderDecoder model(
          "encdec", src_vocab.size(), trg_vocab.size(),
          embed_size, hidden_size, dropout_rate);
      std::cout << "done." << std::endl;

      std::cout << "Initializing trainer ... " << std::flush;
      primitiv::trainers::Adam trainer;
      trainer.set_weight_decay(1e-6);
      trainer.set_gradient_clipping(5);
      model.register_training(trainer);
      std::cout << "done." << std::endl;

      std::cout << "Saving initial model ... " << std::flush;
      ::make_directory(model_dir);
      ::save_value(model_dir + "/batch_size", batch_size);
      ::save_all(
          ::get_model_dir(model_dir, 0), model, trainer,
          1e10, 1e10, std::vector<std::string>(dev_corpus.samples_size()));
      std::cout << "done." << std::endl;

      std::cout << "Start training." << std::endl;
      ::train(
          0, num_epochs,
          model_dir, model, trg_vocab, trainer, train_sampler, dev_sampler);
      std::cout << "Finished." << std::endl;
  });

  return 0;
}
