#include "config.h"

#include <cstdio>
#include <iostream>
#include <string>

#include <primitiv/primitiv.h>
#ifdef PRIMITIV_NMT_USE_CUDA
#include <primitiv/primitiv_cuda.h>
#endif

#include "encoder_decoder.h"
#include "nmt_utils.h"
#include "utils.h"
#include "vocabulary.h"

int main(int argc, char *argv[]) {
  ::check_args(argc, argv, {
      "(file/in) Source vocabulary file",
      "(file/in) Target vocabulary file",
      "(dir/in) Model directory",
      "(int) Epoch",
#ifdef PRIMITIV_NMT_USE_CUDA
      "(int) GPU ID",
#endif
  });

  ::global_try_block([&]() {
      const std::string src_vocab_file = *++argv;
      const std::string trg_vocab_file = *++argv;
      const std::string model_dir = *++argv;
      const unsigned epoch = std::stoi(*++argv);
#ifdef PRIMITIV_NMT_USE_CUDA
      const unsigned gpu_id = std::stoi(*++argv);
#endif

      const std::string subdir = ::get_model_dir(model_dir, epoch);

      const ::Vocabulary src_vocab(src_vocab_file);
      const ::Vocabulary trg_vocab(trg_vocab_file);
      const unsigned bos_id = trg_vocab.stoi("<bos>");
      const unsigned eos_id = trg_vocab.stoi("<eos>");

#ifdef PRIMITIV_NMT_USE_CUDA
      primitiv::devices::CUDA dev(gpu_id);
#else
      primitiv::devices::Eigen dev;
#endif
      primitiv::Device::set_default(dev);

      ::EncoderDecoder<primitiv::Tensor> model;
      model.load(subdir + "/model");

      std::string line;

      while (std::getline(std::cin, line)) {
        const std::vector<unsigned> src_ids = src_vocab.line_to_ids(
            "<bos> " + line + " <eos>");
        if (src_ids.size() < 3) {
          std::cerr << "WARNING: empty sentence" << std::endl;
          std::cout << std::endl;
        }

        std::vector<std::vector<unsigned>> src_batch;
        src_batch.reserve(src_ids.size());
        for (unsigned src_id : src_ids) {
          src_batch.emplace_back(std::vector<unsigned> {src_id});
        }

        const ::Result ret = ::infer_sentence(
            model, bos_id, eos_id, src_batch, 64);
        const std::string hyp_str = ::make_hyp_str(ret, trg_vocab);

        for (unsigned i = 0; i < ret.atten_probs.size(); ++i) {
          std::cout << "a" << (i + 1) << "\t[";
          for (float ap : ret.atten_probs[i]) {
            if (ap > 0.1f) std::printf(" .%01d", static_cast<int>(ap * 10));
            else std::cout << "   ";
          }
          std::cout << " ]" << std::endl;
        }
        std::cout << "h\t" << hyp_str << std::endl;
      }
  });

  return 0;
}
