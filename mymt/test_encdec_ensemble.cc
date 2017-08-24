#include "config.h"

#include <cstdio>
#include <iostream>
#include <memory>
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
      "(dir/in) Model directories (colon-separated)",
      "(int) Epochs (colon-separated)",
#ifdef MYMT_USE_CUDA
      "(int) GPU IDs (colon-separated)",
#endif
  });

  ::global_try_block([&]() {
      const std::string src_vocab_file = *++argv;
      const std::string trg_vocab_file = *++argv;
      const std::vector<std::string> model_dirs = ::split(*++argv, ':');
      const std::vector<std::string> epoch_strs = ::split(*++argv, ':');
      std::vector<unsigned> epochs;
      for (const auto &s : epoch_strs) epochs.emplace_back(std::stoi(s));
#ifdef MYMT_USE_CUDA
      const std::vector<std::string> gpu_ids_strs = ::split(*++argv, ':');
      std::vector<unsigned> gpu_ids;
      for (const auto &s : gpu_ids_strs) gpu_ids.emplace_back(std::stoi(s));
#endif
      if (model_dirs.size() != epochs.size()) {
        throw std::runtime_error(
            std::string("Invalid model description")
            + ". model_dirs.size: " + std::to_string(model_dirs.size())
            + ", epochs.size: " + std::to_string(epochs.size()));
      }

      std::vector<std::string> subdirs;
      for (unsigned i = 0; i < model_dirs.size(); ++i) {
        subdirs.emplace_back(::get_model_dir(model_dirs[i], epochs[i]));
      }

      const ::Vocabulary src_vocab(src_vocab_file);
      const ::Vocabulary trg_vocab(trg_vocab_file);
      const unsigned bos_id = trg_vocab.stoi("<bos>");
      const unsigned eos_id = trg_vocab.stoi("<eos>");

      std::vector<std::unique_ptr<primitiv::Device>> devs;
#ifdef MYMT_USE_CUDA
      for (unsigned gpu_id : gpu_ids) {
        devs.emplace_back(std::unique_ptr<primitiv::Device>(
              new primitiv::CUDADevice(gpu_id)));
      }
#else
      devs.emplace_back(std::unique_ptr<primitiv::Device>(
            new primitiv::CPUDevice()));
#endif

      std::vector<std::unique_ptr<::AttentionEncoderDecoder>> models;
      for (unsigned i = 0; i < subdirs.size(); ++i) {
        primitiv::Device::set_default_device(*devs[i % devs.size()]);
        models.emplace_back(std::unique_ptr<::AttentionEncoderDecoder>(
              new AttentionEncoderDecoder("encdec", subdirs[i] + "/model.")));
      }

      std::string line;
      const std::vector<std::string> chars {"   ", " ░░", " ▒▒", " ▓▓", " ██"};

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

        const ::Result ret = ::infer_sentence_ensemble2(
            devs, models, bos_id, eos_id, src_batch, 64);
        const std::string hyp_str = ::make_hyp_str(ret, trg_vocab);

        for (unsigned i = 0; i < ret.atten_probs.size(); ++i) {
          std::cout << "a" << (i + 1) << "\t[";
          for (float ap : ret.atten_probs[i]) {
            unsigned idx = static_cast<unsigned>(ap * chars.size());
            std::cout << chars[idx >= chars.size() ? chars.size() - 1 : idx];
          }
          std::cout << " ]" << std::endl;
        }
        std::cout << "h\t" << hyp_str << std::endl;
      }
  });

  return 0;
}
