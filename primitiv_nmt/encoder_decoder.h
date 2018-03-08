#ifndef PRIMITIV_NMT_ENCODER_DECODER_H_
#define PRIMITIV_NMT_ENCODER_DECODER_H_

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include <primitiv/primitiv.h>

#include <primitiv_nmt/affine.h>
#include <primitiv_nmt/attention.h>
#include <primitiv_nmt/lstm.h>
#include <primitiv_nmt/utils.h>

template<typename Var>
class EncoderDecoder : public primitiv::Model {
  primitiv::Parameter psrc_emb_, ptrg_emb_;
  ::LSTM<Var> rnn_fw_, rnn_bw_, rnn_dec_;
  ::Attention<Var> att_;
  ::Affine<Var> aff_fbd_, aff_cdj_, aff_jy_;
  Var d_, j_;
  Var trg_emb_;
  Var dec_c0_;

public:
  // New object.
  EncoderDecoder() {
    add("src_emb", psrc_emb_);
    add("trg_emb", ptrg_emb_);
    add("rnn_fw", rnn_fw_);
    add("rnn_bw", rnn_bw_);
    add("rnn_dec", rnn_dec_);
    add("att", att_);
    add("aff_fbd", aff_fbd_);
    add("aff_cdj", aff_cdj_);
    add("aff_jy", aff_jy_);
  }

  // Initializes parameters.
  void init(
      unsigned src_vocab_size, unsigned trg_vocab_size,
      unsigned embed_size, unsigned hidden_size) {
    namespace I = primitiv::initializers;
    psrc_emb_.init({embed_size, src_vocab_size}, I::Uniform(-0.1, 0.1));
    ptrg_emb_.init({embed_size, trg_vocab_size}, I::Uniform(-0.1, 0.1));
    rnn_fw_.init(embed_size, hidden_size);
    rnn_bw_.init(embed_size, hidden_size);
    rnn_dec_.init(2 * embed_size, hidden_size);
    att_.init(2 * hidden_size, hidden_size, hidden_size);
    aff_fbd_.init(2 * hidden_size, hidden_size);
    aff_cdj_.init(3 * hidden_size, embed_size);
    aff_jy_.init(embed_size, trg_vocab_size);
  }

  // Encodes source batch and initializes decoder states.
  void encode(const std::vector<std::vector<unsigned>> &src_batch) {
    namespace F = primitiv::functions;

    const unsigned src_len = src_batch.size();
    const Var invalid;

    // Source embedding
    const Var src_emb = F::parameter<Var>(psrc_emb_);
    std::vector<Var> e_list;
    for (const auto &x : src_batch) {
      e_list.emplace_back(F::pick(src_emb, x, 1));
    }

    // Forward encoding
    rnn_fw_.reset(invalid, invalid);
    std::vector<Var> f_list;
    for (const Var &e : e_list) {
      f_list.emplace_back(rnn_fw_.forward(e));
    }

    // Backward encoding
    rnn_bw_.reset(invalid, invalid);
    std::vector<Var> b_list;
    for (auto it = e_list.rbegin(); it != e_list.rend(); ++it) {
      b_list.emplace_back(rnn_bw_.forward(*it));
    }
    std::reverse(b_list.begin(), b_list.end());

    // Preparing decoder states
    aff_fbd_.reset();
    aff_cdj_.reset();
    aff_jy_.reset();
    const Var last_fb = F::concat({rnn_fw_.get_c(), rnn_bw_.get_c()}, 0);
    dec_c0_ = aff_fbd_.forward(last_fb);

    // Making matrix for calculating attention
    std::vector<Var> fb_list;
    for (unsigned i = 0; i < src_len; ++i) {
      fb_list.emplace_back(F::concat({f_list[i], b_list[i]}, 0));
    }
    att_.reset(fb_list);

    // Other parameters
    trg_emb_ = F::parameter<Var>(ptrg_emb_);
  }

  // Initializes decoder states
  void init_decoder() {
    rnn_dec_.reset(dec_c0_, Var());
    j_ = primitiv::functions::zeros<Var>({embed_size()});
  }

  // Calculates next attention probabilities
  Var decode_atten(const std::vector<unsigned> &trg_words) {
    namespace F = primitiv::functions;
    const Var e = F::pick(trg_emb_, trg_words, 1);
    d_ = rnn_dec_.forward(F::concat({e, j_}, 0));
    return att_.get_probs(d_);
  }

  // Calculates next words
  Var decode_word(const Var &att_probs) {
    namespace F = primitiv::functions;
    const Var c = att_.get_context(att_probs);
    j_ = F::tanh(aff_cdj_.forward(F::concat({c, d_}, 0)));
    return aff_jy_.forward(j_);
  }

  // Calculates the loss function.
  Var loss(const std::vector<std::vector<unsigned>> &trg_batch) {
    namespace F = primitiv::functions;

    std::vector<Var> losses;
    for (unsigned i = 0; i < trg_batch.size() - 1; ++i) {
      const Var att_probs = decode_atten(trg_batch[i]);
      const Var y = decode_word(att_probs);
      losses.emplace_back(F::softmax_cross_entropy(y, trg_batch[i + 1], 0));
    }
    return F::batch::mean(F::sum(losses));
  }

  // Retrieves hyperparameters.
  unsigned src_vocab_size() const { return psrc_emb_.shape()[1]; }
  unsigned trg_vocab_size() const { return ptrg_emb_.shape()[1]; }
  unsigned embed_size() const { return rnn_fw_.input_size(); }
  unsigned hidden_size() const { return rnn_fw_.output_size(); }
};

#endif  // PRIMITIV_NMT_ENCODER_DECODER_H_
