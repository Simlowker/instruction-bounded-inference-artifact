#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

static std::string repeat_text(const char * base, int target_tokens) {
    std::string out;
    for (int i = 0; i < target_tokens * 4; i++) {
        out += base;
        out += ' ';
    }
    return out;
}

struct BenchResult {
    std::string model;
    std::string quant;
    int prompt_len;
    int tokens_generated;
    double tokens_per_sec;
    double time_ms;
    double prefill_tok_per_sec;
    double prefill_ms;
};

static bool run_bench(const char * model_path, const char * model_name, const char * quant,
                      int target_prompt_len, int n_gen, std::vector<BenchResult> & results) {
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    auto * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "SKIP: cannot load %s\n", model_path);
        return false;
    }

    const auto * vocab = llama_model_get_vocab(model);

    auto cparams = llama_context_default_params();
    cparams.n_ctx    = 2048;
    cparams.n_batch  = 2048;
    cparams.n_threads = 0; // auto
    cparams.no_perf  = false;

    auto * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "SKIP: cannot create context for %s\n", model_path);
        llama_model_free(model);
        return false;
    }

    // Build prompt text — overshoot then trim tokens to target
    std::string prompt_text = repeat_text(
        "The quick brown fox jumps over the lazy dog.",
        target_prompt_len);

    // First pass: determine how many tokens we get
    int needed = llama_tokenize(vocab, prompt_text.c_str(), prompt_text.size(),
                                nullptr, 0, true, false);
    if (needed == 0) {
        fprintf(stderr, "SKIP: empty tokenization for %s\n", model_path);
        llama_free(ctx);
        llama_model_free(model);
        return false;
    }
    int buf_size = (needed < 0) ? -needed : needed;
    std::vector<llama_token> tokens(buf_size + 16);
    int n_tokens = llama_tokenize(vocab, prompt_text.c_str(), prompt_text.size(),
                                  tokens.data(), tokens.size(), true, false);
    if (n_tokens < 0) {
        fprintf(stderr, "SKIP: tokenization failed for %s (need %d)\n", model_path, -n_tokens);
        llama_free(ctx);
        llama_model_free(model);
        return false;
    }

    // Trim to target
    if (n_tokens > target_prompt_len) n_tokens = target_prompt_len;
    tokens.resize(n_tokens);

    // Reset perf counters
    llama_perf_context_reset(ctx);

    // Prefill
    auto batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "SKIP: prefill failed for %s prompt=%d\n", model_path, target_prompt_len);
        llama_free(ctx);
        llama_model_free(model);
        return false;
    }

    // Sampler
    auto sparams = llama_sampler_chain_default_params();
    auto * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // Generation loop — ignore EOS to get consistent generation counts
    int generated = 0;

    for (int i = 0; i < n_gen; i++) {
        llama_token new_token = llama_sampler_sample(smpl, ctx, -1);

        generated++;
        auto one = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx, one) != 0) break;
    }

    // Collect perf
    auto perf = llama_perf_context(ctx);

    double gen_tok_per_sec = (perf.t_eval_ms > 0)
        ? (double)perf.n_eval / (perf.t_eval_ms / 1000.0) : 0;
    double prefill_tok_per_sec = (perf.t_p_eval_ms > 0)
        ? (double)perf.n_p_eval / (perf.t_p_eval_ms / 1000.0) : 0;
    double total_ms = perf.t_p_eval_ms + perf.t_eval_ms;

    fprintf(stderr, "  %-40s prompt=%4d  gen=%3d  prefill=%.1f tok/s  gen=%.1f tok/s  total=%.0f ms\n",
            model_name, perf.n_p_eval, perf.n_eval,
            prefill_tok_per_sec, gen_tok_per_sec, total_ms);

    results.push_back({model_name, quant, perf.n_p_eval, perf.n_eval,
                       gen_tok_per_sec, total_ms, prefill_tok_per_sec, perf.t_p_eval_ms});

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <models_dir> [n_gen=64]\n", argv[0]);
        return 1;
    }

    std::string models_dir = argv[1];
    int n_gen = (argc >= 3) ? atoi(argv[2]) : 64;

    llama_backend_init();

    struct ModelInfo {
        const char * subpath;
        const char * name;
        const char * quant;
    };

    ModelInfo models[] = {
        {"SmolLM2-135M/smollm2-135m-Q4_0.gguf",           "SmolLM2-135M",           "Q4_0"},
        {"Qwen/qwen2.5-0.5b-Q4_0.gguf",                   "Qwen2.5-0.5B",           "Q4_0"},
        {"Pythia-70M/pythia-70m-Q4_0.gguf",                "Pythia-70M",             "Q4_0"},
        {"GPT2-124M/gpt2.Q8_0.gguf",                      "GPT2-124M",              "Q8_0"},
        {"SmolLM2-135M/smollm2-135m-instruct-q8_0.gguf",  "SmolLM2-135M-instruct",  "Q8_0"},
    };

    int prompt_lens[] = {16, 128, 512};

    std::vector<BenchResult> results;

    for (auto & m : models) {
        std::string path = models_dir + "/" + m.subpath;
        fprintf(stderr, "\n=== %s (%s) ===\n", m.name, m.quant);

        for (int pl : prompt_lens) {
            run_bench(path.c_str(), m.name, m.quant, pl, n_gen, results);
        }
    }

    // CSV output
    const char * csv_path = "results.csv";
    FILE * f = fopen(csv_path, "w");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", csv_path);
        return 1;
    }

    fprintf(f, "model,quant,prompt_len,tokens_generated,tokens_per_sec,time_ms,prefill_tok_per_sec,prefill_ms\n");
    for (auto & r : results) {
        fprintf(f, "%s,%s,%d,%d,%.2f,%.2f,%.2f,%.2f\n",
                r.model.c_str(), r.quant.c_str(),
                r.prompt_len, r.tokens_generated,
                r.tokens_per_sec, r.time_ms,
                r.prefill_tok_per_sec, r.prefill_ms);
    }
    fclose(f);
    fprintf(stderr, "\nResults written to %s\n", csv_path);

    llama_backend_free();
    return 0;
}
