#include "config.h"

#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include <primitiv/primitiv.h>
#ifdef MYMT_USE_CUDA
#include <primitiv/primitiv_cuda.h>
#endif

#include "fixed_length_decoder.h"
#include "fld_utils.h"
#include "utils.h"
#include "vocabulary.h"

int main(int argc, char *argv[]) {
  ::check_args(argc, argv, {
      "(file/in) Source vocabulary file",
      "(file/in) Target vocabulary file",
      "(dir/in) Model directory",
      "(int) Epoch",
#ifdef MYMT_USE_CUDA
      "(int) GPU ID",
#endif
  });

  ::global_try_block([&]() {
      const std::string src_vocab_file = *++argv;
      const std::string trg_vocab_file = *++argv;
      const std::string model_dir = *++argv;
      const unsigned epoch = std::stoi(*++argv);
#ifdef MYMT_USE_CUDA
      const unsigned gpu_id = std::stoi(*++argv);
#endif

      const std::string subdir = ::get_model_dir(model_dir, epoch);

      const ::Vocabulary src_vocab(src_vocab_file);
      const ::Vocabulary trg_vocab(trg_vocab_file);

#ifdef MYMT_USE_CUDA
      primitiv::CUDADevice dev(gpu_id);
#else
      primitiv::CPUDevice dev;
#endif
      primitiv::Device::set_default_device(dev);

      ::FixedLengthDecoder<primitiv::Tensor> model("fld", subdir + "/model.");

      std::string line;

      while (std::getline(std::cin, line)) {
        const std::vector<unsigned> src_ids = src_vocab.line_to_ids(
            "<bos> " + line + " <eos>");
        std::vector<std::vector<unsigned>> src_batch;
        src_batch.reserve(src_ids.size());
        for (unsigned i = 0; i < src_ids.size(); ++i) {
          src_batch.emplace_back(std::vector<unsigned> {src_ids[i]});
        }

        std::getline(std::cin, line);
        const std::vector<unsigned> trg_ids = trg_vocab.line_to_ids(
            "<bos> " + line + " <eos>");
        std::vector<std::vector<unsigned>> trg_batch;
        trg_batch.reserve(trg_ids.size());
        for (unsigned i = 0; i < trg_ids.size(); ++i) {
          trg_batch.emplace_back(std::vector<unsigned> {trg_ids[i]});
        }

        model.encode(src_batch, false);
        const float score = model.loss(trg_batch, false).to_vector()[0];
        std::cout << score << std::endl;
      }
  });

  return 0;
}
