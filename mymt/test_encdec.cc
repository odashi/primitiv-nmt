#include "config.h"

#include <iostream>
#include <string>

#include <primitiv/primitiv.h>
#ifdef MYMT_USE_CUDA
#include <primitiv/primitiv_cuda.h>
#endif

#include "attention_encoder_decoder.h"
#include "nmt_utils.h"
#include "utils.h"
#include "vocabulary.h"

int main(int argc, char *argv[]) {
  ::check_args(argc, argv, {
      "(file/in) Source vocabulary file",
      "(file/in) Target vocabulary file",
      "(dir/in) Model directory",
      "(int) Epoch",
  });

  ::global_try_block([&]() {
      const std::string src_vocab_file = *++argv;
      const std::string trg_vocab_file = *++argv;
      const std::string model_dir = *++argv;
      const unsigned epoch = std::stoi(*++argv);
      const std::string subdir = ::get_model_dir(model_dir, epoch);

      const ::Vocabulary src_vocab(src_vocab_file);
      const ::Vocabulary trg_vocab(trg_vocab_file);

#ifdef MYMT_USE_CUDA
      primitiv::CUDADevice dev(0);
#else
      primitiv::CPUDevice dev;
#endif
      primitiv::Device::set_default_device(dev);

      ::AttentionEncoderDecoder model("encdec", subdir + "/model.");

      std::string line;
      while (std::getline(std::cin, line)) {
        const std::vector<unsigned> src_ids = src_vocab.line_to_ids(
            "<bos> " + line + " <eos>");
        std::vector<std::vector<unsigned>> src_batch;
        src_batch.reserve(src_ids.size());
        for (unsigned src_id : src_ids) {
          src_batch.emplace_back(std::vector<unsigned> {src_id});
        }
        const std::string trg_sent = ::infer_sentence(
            model, trg_vocab, src_batch, 64);
        std::cout << trg_sent << std::endl;
      }
  });

  return 0;
}
