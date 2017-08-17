#include "config.h"

#include <cstdio>
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
      const unsigned bos_id = trg_vocab.stoi("<bos>");
      const unsigned eos_id = trg_vocab.stoi("<eos>");

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
        const ::Result ret = ::infer_sentence(
            model, bos_id, eos_id, src_batch, 64);
        const std::string hyp_str = ::make_hyp_str(ret, trg_vocab);
        for (unsigned i = 0; i < ret.atten_probs.size(); ++i) {
          std::cout << "a" << (i + 1) << '\t';
          for (float ap : ret.atten_probs[i]) std::printf(" %.2f", ap);
          std::cout << std::endl;
        }
        std::cout << "h\t" << hyp_str << std::endl;
      }
  });

  return 0;
}
