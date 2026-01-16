// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>

#include "gtest/gtest.h"
#include "ort_genai.h"

#if __cplusplus >= 202002L
#define USE_SPAN
#endif

struct OgaObjects {
  OgaModel* model;
  OgaTokenizer* tokenizer;
  OgaTokenizerStream* tokenizer_stream;
};

class OgaEnvironment : public ::testing::Environment {
 public:
  explicit OgaEnvironment(const char* model_path) {
    model_ = OgaModel::Create(model_path);
    tokenizer_ = OgaTokenizer::Create(*model_);
    tokenizer_stream_ = OgaTokenizerStream::Create(*tokenizer_);
  }

  ~OgaEnvironment() override = default;

  static OgaObjects getOgaObjects() {
    return {model_.get(), tokenizer_.get(), tokenizer_stream_.get()};
  }

 private:
  inline static std::unique_ptr<OgaModel> model_;
  inline static std::unique_ptr<OgaTokenizer> tokenizer_;
  inline static std::unique_ptr<OgaTokenizerStream> tokenizer_stream_;
};

/**
 * @brief This test the kv cache reuse and rewind API.
 * You can rewind to any position you want to and ask question again.
 * Below example shows:
 *  after Q1 A1 Q2 A2 Q3 A3,
 *  rewind to Q1 A1, and ask Q2 again,
 *  you can see new_A2 is just exact same as A2.
 */
TEST(HybridLlmOga, KVcacheReuse) {
  bool verbose = false;

  auto oga_objects = OgaEnvironment::getOgaObjects();
  auto model = oga_objects.model;
  auto tokenizer = oga_objects.tokenizer;
  auto tokenizer_stream = oga_objects.tokenizer_stream;

  std::vector<std::string> text{
    "My name is Alex, I work as a data scientist at Orbital AI.",
    "I live in Berlin and enjoy playing the piano in my free time.",
    "Where do I live and what do I enjoy doing?"
  };

  if (verbose) {
    std::cout << "test_kvcache_reuse\n";
  }
  int totalL = 100;
  int total = 0;
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 4096);
  params->SetSearchOption("min_length", 20);

  std::unique_ptr<OgaSequences> seq;
  std::vector<std::vector<int32_t>> ids_in(text.size());
  std::vector<std::vector<int32_t>> ids_out(text.size());
  std::vector<int32_t> ids_out_rewind;
  for (int i = 0; i < (int)text.size(); i++) {
    seq = OgaSequences::Create();
    std::string ts = std::string("<|user|>") + text[i] + "<|end|><|assistant|>";
    tokenizer->Encode(ts.c_str(), *seq);
    ids_in[i].resize(seq->SequenceCount(0));
    memcpy(
      ids_in[i].data(), seq->SequenceData(0),
      seq->SequenceCount(0) * sizeof(int32_t)
    );
  }
  if (verbose) {
    std::cout << "input tokens length:  ";
    for (int j = 0; j < (int)text.size(); j++) {
      std::cout << ids_in[j].size() << " ";
    }
    std::cout << "\n";
  }
  size_t rewind_pos = 0;

  auto generator = OgaGenerator::Create(*model, *params);
#ifdef USE_SPAN
  std::span<const int32_t> oseq;
  size_t oseq_size = 0;
#else
  const int32_t* oseq;
  size_t oseq_size = 0;
#endif
  for (int j = 0; j < (int)text.size(); j++) {
    total = 0;
    generator->AppendTokens(ids_in[j].data(), ids_in[j].size());
    while (!generator->IsDone()) {
      generator->GenerateNextToken();
      const auto num_tokens = generator->GetSequenceCount(0);
      const auto new_token = generator->GetSequenceData(0)[num_tokens - 1];
#ifdef USE_SPAN
      oseq = generator->GetSequence(0);
      oseq_size = oseq.size();
#else
      oseq = generator->GetSequenceData(0);
      oseq_size = generator->GetSequenceCount(0);
#endif
      ids_out[j].emplace_back(new_token);
      auto decoded_answer = tokenizer_stream->Decode(new_token);
      if (verbose) {
        std::cout << decoded_answer << std::flush;
      }
      total++;
      if (total >= totalL) break;
    }
    if (verbose) {
      std::cout << "\n====END Answer " << j << " ===== LenIn/LenOut/LenAll "
                << ids_in[j].size() << " " << ids_out[j].size() << " "
                << oseq_size << "\n\n";
    }
  }
  if (verbose) {
    std::cout << "test rewind to the Q1+A1 position\n";
  }
  rewind_pos = ids_in[0].size() + ids_out[0].size();
  generator->RewindTo(rewind_pos);
  generator->AppendTokens(ids_in[1].data(), ids_in[1].size());
  total = 0;
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
    const auto num_tokens = generator->GetSequenceCount(0);
    const auto new_token = generator->GetSequenceData(0)[num_tokens - 1];
#ifdef USE_SPAN
    oseq = generator->GetSequence(0);
#else
    oseq = generator->GetSequenceData(0);
    oseq_size = generator->GetSequenceCount(0);
#endif
    ids_out_rewind.emplace_back(new_token);
    auto decoded_answer = tokenizer_stream->Decode(new_token);
    if (verbose) {
      std::cout << decoded_answer << std::flush;
    }
    total++;
    if (total >= totalL) break;
  }
  EXPECT_EQ(ids_out_rewind, ids_out[1]);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  if (argc != 2) {
    throw std::runtime_error("Usage: test_rewind [gtest_args] <model_path>");
  }

  ::testing::AddGlobalTestEnvironment(new OgaEnvironment(argv[1]));

  return RUN_ALL_TESTS();
}
