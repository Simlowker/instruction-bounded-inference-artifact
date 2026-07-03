// Embedding endpoint for on-chain vector generation
// Uses the already-loaded model to produce embeddings via llama_encode()

#include "embed.h"
#include "auth.h"
#include "http.h"
#include "main_.h"
#include "ready.h"
#include "utils.h"

#include "llama.h"

#include <string>
#include <vector>

#include "ic_api.h"

// ICPP-PROFILING: ICP instruction counter (same as main_.cpp)
#ifdef __wasm__
__attribute__((import_module("ic0")))
__attribute__((import_name("performance_counter")))
extern "C" uint64_t icpp_embed_perf_counter(uint32_t counter_type);
#else
static inline uint64_t icpp_embed_perf_counter(uint32_t t) { (void)t; return 0; }
#endif
#define EMBED_PERF() icpp_embed_perf_counter(0)

void embed() {
  IC_API ic_api(CanisterUpdate{std::string(__func__)}, false);
  uint64_t perf_start = EMBED_PERF();
  std::string error_msg;

  if (!has_admin_update_or_whitelisted(ic_api)) {
    ic_api.to_wire(CandidTypeVariant{
        "Err", CandidTypeVariant{"Other", CandidTypeText{"Access denied"}}});
    return;
  }

  if (!ready_for_inference) {
    ic_api.to_wire(CandidTypeVariant{
        "Err",
        CandidTypeVariant{"Other", CandidTypeText{"Model not loaded"}}});
    return;
  }

  llama_context *ctx = icpp_get_ctx();
  llama_model *model = (g_model && *g_model) ? *g_model : nullptr;

  if (!model || !ctx) {
    ic_api.to_wire(CandidTypeVariant{
        "Err", CandidTypeVariant{
                   "Other", CandidTypeText{"Model or context not initialized"}}});
    return;
  }

  // Read input text from wire
  std::string input_text;
  CandidTypeRecord r_in;
  r_in.append("text", CandidTypeText{&input_text});
  ic_api.from_wire(r_in);

  if (input_text.empty()) {
    ic_api.to_wire(CandidTypeVariant{
        "Err", CandidTypeVariant{"Other", CandidTypeText{"Empty input text"}}});
    return;
  }

  // Enable embedding mode on the context
  llama_set_embeddings(ctx, true);

  // Tokenize input using llama C API
  const llama_vocab *vocab = llama_model_get_vocab(model);
  int max_tokens = input_text.size() + 16;
  std::vector<llama_token> tokens(max_tokens);
  int n = llama_tokenize(vocab, input_text.c_str(), input_text.size(),
                         tokens.data(), max_tokens, true, true);
  if (n < 0) {
    tokens.resize(-n);
    n = llama_tokenize(vocab, input_text.c_str(), input_text.size(),
                       tokens.data(), -n, true, true);
  }
  if (n > 0) {
    tokens.resize(n);
  }

  if (tokens.empty()) {
    ic_api.to_wire(CandidTypeVariant{
        "Err",
        CandidTypeVariant{"Other", CandidTypeText{"Tokenization failed"}}});
    return;
  }

  // Clear the memory/KV cache for a clean encode
  llama_memory_t mem = llama_get_memory(ctx);
  if (mem) {
    llama_memory_clear(mem, true);
  }

  // Prepare batch
  llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
  for (size_t i = 0; i < tokens.size(); i++) {
    batch.token[i] = tokens[i];
    batch.pos[i] = i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = (i == tokens.size() - 1) ? 1 : 0;
  }
  batch.n_tokens = tokens.size();

  // Encode (forward pass for embeddings)
  int ret = llama_encode(ctx, batch);
  llama_batch_free(batch);

  if (ret != 0) {
    ic_api.to_wire(CandidTypeVariant{
        "Err", CandidTypeVariant{
                   "Other",
                   CandidTypeText{"llama_encode failed with code " +
                                  std::to_string(ret)}}});
    return;
  }

  // Get embeddings — try pooled first (seq_id=0), then positional
  int n_embd = llama_model_n_embd(model);
  float *emb = llama_get_embeddings_seq(ctx, 0);
  if (!emb) {
    emb = llama_get_embeddings_ith(ctx, -1);
  }

  if (!emb) {
    ic_api.to_wire(CandidTypeVariant{
        "Err", CandidTypeVariant{
                   "Other", CandidTypeText{"Failed to get embeddings"}}});
    return;
  }

  // Normalize the embedding vector (L2 norm)
  float norm = 0.0f;
  for (int i = 0; i < n_embd; i++) {
    norm += emb[i] * emb[i];
  }
  norm = sqrtf(norm);

  // Build output: return as vec float32
  std::vector<double> embedding_vec;
  embedding_vec.reserve(n_embd);
  for (int i = 0; i < n_embd; i++) {
    embedding_vec.push_back(
        static_cast<double>(norm > 0.0f ? emb[i] / norm : emb[i]));
  }

  // Build profiling JSON
  uint64_t perf_end = EMBED_PERF();
  uint64_t total_instructions = perf_end - perf_start;
  std::string profiling_json = "{\"total\":" + std::to_string(total_instructions) +
                               ",\"n_tokens\":" + std::to_string(tokens.size()) +
                               ",\"n_embd\":" + std::to_string(n_embd) + "}";

  // Return over the wire
  CandidTypeRecord r_out;
  r_out.append("status_code", CandidTypeNat16{Http::StatusCode::OK});
  r_out.append("n_embd", CandidTypeNat32{static_cast<uint32_t>(n_embd)});
  r_out.append("n_tokens", CandidTypeNat32{static_cast<uint32_t>(tokens.size())});
  r_out.append("embeddings", CandidTypeVecFloat64{embedding_vec});
  r_out.append("profiling", CandidTypeText{profiling_json});
  ic_api.to_wire(CandidTypeVariant{"Ok", r_out});
}
