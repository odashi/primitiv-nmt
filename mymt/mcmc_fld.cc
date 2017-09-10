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

#include "attention_encoder_decoder.h"
#include "fixed_length_decoder.h"
#include "fld_utils.h"
#include "utils.h"
#include "vocabulary.h"

int main(int argc, char *argv[]) {
  ::check_args(argc, argv, {
      "(file/in) Source vocabulary file",
      "(file/in) Target vocabulary file",
      "(dir/in) FLD model directory",
      "(int) FLD epoch",
      "(dir/in) AttEncDec model directory",
      "(int) AttEncDec epoch",
#ifdef MYMT_USE_CUDA
      "(int) GPU ID",
#endif
  });

  ::global_try_block([&]() {
      const std::string src_vocab_file = *++argv;
      const std::string trg_vocab_file = *++argv;
      const std::string fld_model_dir = *++argv;
      const unsigned fld_epoch = std::stoi(*++argv);
      const std::string encdec_model_dir = *++argv;
      const unsigned encdec_epoch = std::stoi(*++argv);
#ifdef MYMT_USE_CUDA
      const unsigned gpu_id = std::stoi(*++argv);
#endif

      const std::string fld_subdir = ::get_model_dir(fld_model_dir, fld_epoch);
      const std::string encdec_subdir = ::get_model_dir(encdec_model_dir, encdec_epoch);

      const ::Vocabulary src_vocab(src_vocab_file);
      const ::Vocabulary trg_vocab(trg_vocab_file);
      const unsigned bos_id = trg_vocab.stoi("<bos>");
      const unsigned eos_id = trg_vocab.stoi("<eos>");

#ifdef MYMT_USE_CUDA
      primitiv::CUDADevice dev(gpu_id);
#else
      primitiv::CPUDevice dev;
#endif
      primitiv::Device::set_default_device(dev);

      ::FixedLengthDecoder<primitiv::Tensor> model("fld", fld_subdir + "/model.");
      ::AttentionEncoderDecoder<primitiv::Tensor> proposal("encdec", encdec_subdir + "/model.");

      std::random_device rd;
      std::mt19937 rng(rd());
      std::uniform_int_distribution<unsigned> dist(2, trg_vocab.size());
      std::uniform_real_distribution<double> unit_dist(0, 1);
      auto gen = [&]() {
        unsigned ret = dist(rng);
        return ret == trg_vocab.size() ? 0 : ret; // |V| -> <unk>
      };
      auto unif = [&]() {
        return unit_dist(rng);
      };

      std::string line;

      while (std::getline(std::cin, line)) {
        const std::vector<std::string> src_words = ::split(line);
        if (src_words.size() < 3) {
          throw std::runtime_error("Format: w1 w2 ... wn trg_len num_samples");
        }
        const unsigned trg_len = std::stoi(src_words[src_words.size() - 2]);
        if (trg_len < 1) {
          throw std::runtime_error("trg_len should be greater than 0.");
        }
        const unsigned num_samples = std::stoi(src_words.back());

        std::vector<std::vector<unsigned>> src_batch;
        src_batch.reserve(src_words.size());
        src_batch.emplace_back(std::vector<unsigned> {bos_id});
        for (unsigned i = 0; i < src_words.size(); ++i) {
          src_batch.emplace_back(
              std::vector<unsigned> {src_vocab.stoi(src_words[i])});
        }
        src_batch.emplace_back(std::vector<unsigned> {eos_id});

        std::cout << "source:";
        for (const auto &v : src_batch) {
          std::cout << ' ' << src_vocab.itos(v[0]);
        }
        std::cout << std::endl;
        std::cout << "trg_len: " << trg_len << std::endl;
        std::cout << "num_samples: " << num_samples << std::endl;

        model.encode(src_batch, false);
        proposal.encode(src_batch, false);

        // Initial sample
        std::vector<std::vector<unsigned>> trg_batch;
        trg_batch.reserve(trg_len + 2);
        trg_batch.emplace_back(std::vector<unsigned> {bos_id});
        for (unsigned i = 0; i < trg_len; ++i) {
          trg_batch.emplace_back(std::vector<unsigned> {gen()});
        }
        trg_batch.emplace_back(std::vector<unsigned> {eos_id});

        // Initial score
        float trg_score = model.loss(trg_batch, false).to_vector()[0];

        std::cout << "target-0:";
        for (const auto &v : trg_batch) {
          std::cout << ' ' << trg_vocab.itos(v[0]);
        }
        std::cout << "\tscore-0: " << trg_score << std::endl;

        for (unsigned n = 1; n <= num_samples; ++n) {
          // New sample
          std::vector<std::vector<unsigned>> new_trg_batch = trg_batch;

          double lqx = 0, lqy = 0;

          if (unif() < 0.4) {
            unsigned id1 = static_cast<unsigned>(unif() * trg_len);
            unsigned id2 = 1 + static_cast<unsigned>(unif() * trg_len);
            while (id2 == id1) id2 = 1 + static_cast<unsigned>(unif() * trg_len);
            if (id1 > id2) {
              const unsigned temp = id1;
              id1 = id2;
              id2 = temp;
            }

            proposal.init_decoder(false);
            for (unsigned i = 0; i < id1; ++i) {
              const auto ap = proposal.decode_atten(new_trg_batch[i], false);
              const auto wp = proposal.decode_word(ap, false);
            }
            for (unsigned i = id1; i < id2; ++i) {
              namespace F = primitiv::operators;
              const auto ap = proposal.decode_atten(new_trg_batch[i], false);
              const auto wp = F::log_softmax(proposal.decode_word(ap, false), 0);
              lqx += wp.to_vector()[new_trg_batch[i + 1][0]];
            }

            proposal.init_decoder(false);
            unsigned prev_id = new_trg_batch[id1][0];
            for (unsigned i = 0; i < id1; ++i) {
              const auto ap = proposal.decode_atten(new_trg_batch[i], false);
              const auto wp = proposal.decode_word(ap, false);
            }
            for (unsigned i = id1; i < id2; ++i) {
              namespace F = primitiv::operators;
              const auto ap = proposal.decode_atten({prev_id}, false);
              const auto wp = F::log_softmax(proposal.decode_word(ap, false), 0);
              const auto noise = F::random::gumbel<primitiv::Tensor>(wp.shape(), 0, 0.1);
              const unsigned new_id = ::argmax((wp + noise).to_vector());
              lqy += wp.to_vector()[new_id];
              new_trg_batch[i + 1][0] = new_id;
              prev_id = new_id;
            }
          } else if (unif() < 0.8) {
            const unsigned change_id = static_cast<unsigned>(unif() * trg_len);
            const auto sample_ret = model.sample(trg_batch, change_id);
            new_trg_batch[change_id + 1][0] = sample_ret.new_id;
            lqx = sample_ret.org_score;
            lqy = sample_ret.new_score;
          } else {
            if (trg_len >= 2) {
              unsigned span = 2 + static_cast<unsigned>(unif() * (trg_len - 2));
              unsigned l = static_cast<unsigned>(unif() * (trg_len - span));
              unsigned m = l + 1 + static_cast<unsigned>(unif() * (span - 2));
              unsigned r = l + span;
              std::reverse(new_trg_batch.begin() + l + 1, new_trg_batch.begin() + m + 1);
              std::reverse(new_trg_batch.begin() + m + 1, new_trg_batch.begin() + r + 1);
              std::reverse(new_trg_batch.begin() + l + 1, new_trg_batch.begin() + r + 1);
            }
          }

          const double lpx = -trg_score;
          const double lpy = -model.loss(new_trg_batch, false).to_vector()[0];
          const double alpha = std::exp(lpy + lqx - lpx - lqy);

          std::cout << "lp(x): " << lpx << std::endl;
          std::cout << "lp(y): " << lpy << std::endl;
          std::cout << "lq(x|y): " << lqx << std::endl;
          std::cout << "lq(y|x): " << lqy << std::endl;
          std::cout << "alpha: " << alpha << std::endl;
          if (unif() <= alpha) {
            trg_batch = new_trg_batch;
            trg_score = -lpy;
          }

          std::cout << "target-" << n << ':';
          for (const auto &v : trg_batch) {
            std::cout << ' ' << trg_vocab.itos(v[0]);
          }
          std::cout << "\tscore-" << n << ": " << trg_score << std::endl;
        }
      }
  });

  return 0;
}
