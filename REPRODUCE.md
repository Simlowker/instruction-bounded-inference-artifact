# REPRODUCE.md — End-to-end reproduction of the 2.9× fork gap

This document is the single-source procedure to reproduce the headline mainnet result on a local dfx replica.

**Time budget:** ~60 min (15 min build per WASM, 5–10 min model upload per fork, 5 min bench).
**Cost:** zero ICP cycles required. Local dfx fabricates cycles for free.
**Disk:** ~50 GB free recommended (build artifacts + dfx state).

## 0. Prerequisites

- macOS or Linux
- `dfx` (DFINITY Canister SDK) ≥ 0.31.0 — https://internetcomputer.org/docs/current/developer-docs/setup/install/
- `icpp-pro` ≥ 5.3.1 (Python package) — `pip install icpp-pro`
- `wasi-sdk` 25.0 (installed automatically by icpp on first build)
- `wasm-opt` (Binaryen) — `brew install binaryen` (macOS) or `apt install binaryen` (Linux)
- Python ≥ 3.11

## 1. Clone the two sources

```bash
# Comoto fork (this work)
git clone https://github.com/Simlowker/gian-paper-artifact-source.git gian
cd gian/llama_cpp_canister
# The repo is a byte-exact export of Simlowker/gian@8cda13b (tag: paper-v14-bench-snapshot).
# The llama.cpp fork sources are already vendored in src/llama_cpp_onicai_fork/ — no extra clone needed.

# onicai baseline (latest tagged release)
cd /tmp
git clone --depth 1 --branch v0.10.1 https://github.com/onicai/llama_cpp_canister.git onicai-v0.10.1
cd onicai-v0.10.1/src
git clone https://github.com/onicai/llama_cpp_onicai_fork.git
```

## 2. Build both WASMs

### Comoto fork

```bash
cd gian/llama_cpp_canister

# Generate build-info.cpp (the auto-generated file is missing from the working tree
# in some snapshots; if so, create it manually with the LLAMA_BUILD_NUMBER /
# LLAMA_COMMIT / LLAMA_COMPILER / LLAMA_BUILD_TARGET symbols — any values will do
# functionally, only their existence is required by the linker).
cat > src/llama_cpp_onicai_fork/common/build-info.cpp <<'EOF'
int LLAMA_BUILD_NUMBER = 1;
char const *LLAMA_COMMIT = "8cda13b";
char const *LLAMA_COMPILER = "clang (wasi-sdk-25.0)";
char const *LLAMA_BUILD_TARGET = "wasm32-wasi";
EOF

icpp build-wasm
shasum -a 256 build/llama_cpp.wasm
# Expected (rebuild-as-of-2026-05-19): da112d99...
# Note: byte-exact hash will vary with toolchain version; LTO is non-deterministic.
```

### onicai baseline v0.10.1

```bash
cd /tmp/onicai-v0.10.1

make build-info-cpp-wasm
icpp build-wasm
# The icpp post_wasm_function step calls scripts.optimize_wasm.main which
# requires the binaryen Python package. If unavailable, run wasm-opt manually:
wasm-opt -Os --strip-target-features build/llama_cpp.wasm -o build/llama_cpp_opt.wasm
shasum -a 256 build/llama_cpp_opt.wasm
# Expected: b6ccbff0... (varies with toolchain; ~4.6 MB)
```

## 3. Start a local dfx replica and deploy Comoto

```bash
cd gian/llama_cpp_canister
dfx start --background --clean

dfx deploy llama_cpp
dfx canister info llama_cpp
# Module hash on-chain should match da112d99... (or your rebuilt hash)
```

## 4. Download Qwen 2.5 0.5B Instruct Q8_0

```bash
# Source: HuggingFace (public, ungated)
mkdir -p models/Qwen
cd models/Qwen
curl -L -o qwen2.5-0.5b-instruct-q8_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf

shasum -a 256 qwen2.5-0.5b-instruct-q8_0.gguf
# Expected: ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
# Size: 675710816 bytes
```

## 5. Upload Qwen and top-up cycles

```bash
cd gian/llama_cpp_canister

# Top up cycles (local fabricate)
dfx ledger fabricate-cycles --canister llama_cpp --amount 100 --network local

# Upload Qwen
python3 -m scripts.upload --network local --canister llama_cpp \
  --canister-filename models/model.gguf \
  models/Qwen/qwen2.5-0.5b-instruct-q8_0.gguf

# Verify
dfx canister call llama_cpp uploaded_file_details \
  '(record {filename="models/model.gguf"})'
# Expected: filesha256 = "ca59ca7f..."
```

## 6. Bench Comoto: expect 29 tok/call OK, gen@30 TRAP

Run `bash scripts/run_bench.sh gian` (or do it manually below).

```bash
PROMPT='<|im_start|>user\nAnswer the following question as brief as possible. This is the question: What are the key differences between proof-of-work and proof-of-stake consensus mechanisms?<|im_end|>\n<|im_start|>assistant\n'

# Set max tokens = 29 for Comoto
dfx canister call llama_cpp set_max_tokens \
  '(record { max_tokens_query = 1 : nat64; max_tokens_update = 29 : nat64 })'

# Load model
dfx canister call llama_cpp load_model \
  '(record { args = vec {"--model"; "models/model.gguf"} })'

# New chat
dfx canister call llama_cpp new_chat \
  '(record { args = vec {"--model"; "models/model.gguf"; "--prompt-cache"; "prompt.cache"} })'

# Prefill ×2 (until prompt_remaining is empty)
for i in 1 2 3; do
  dfx canister call llama_cpp run_update \
    "(record { args = vec {\"--model\"; \"models/model.gguf\"; \"--prompt-cache\"; \"prompt.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-n\"; \"512\"; \"-p\"; \"$PROMPT\"} })"
done

# Pure gen @ 29 → should return OK with output ending "...where miners try to solve complex"
dfx canister call llama_cpp run_update \
  "(record { args = vec {\"--model\"; \"models/model.gguf\"; \"--prompt-cache\"; \"prompt.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-n\"; \"512\"; \"-p\"; \"$PROMPT\"} })"

# Pure gen @ 30 → should TRAP with IC0522
dfx canister call llama_cpp set_max_tokens \
  '(record { max_tokens_query = 1 : nat64; max_tokens_update = 30 : nat64 })'
dfx canister call llama_cpp run_update \
  "(record { args = vec {\"--model\"; \"models/model.gguf\"; \"--prompt-cache\"; \"prompt.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-n\"; \"512\"; \"-p\"; \"$PROMPT\"} })"
```

Expected output strings:

- Prefill 2 ends with: `"Proof-of-work and proof-of-stake are two different consensus mechanisms used in blockchain technology. Proof"`
- Gen@29 ends with: `"Proof-of-work and proof-of-stake are two different consensus mechanisms used in blockchain technology. Proof-of-work is where miners try to solve complex"`
- Gen@30 TRAPs with: `Canister exceeded the limit of 40000000000 instructions for single message execution (IC0522)`

## 7. Bench onicai: expect 10 tok/call OK, gen@11 TRAP

```bash
# Reinstall canister with onicai WASM (loses Comoto state)
dfx canister install llama_cpp --mode reinstall \
  --wasm /tmp/onicai-v0.10.1/build/llama_cpp_opt.wasm \
  --yes

# Top up cycles again
dfx ledger fabricate-cycles --canister llama_cpp --amount 100 --network local

# Re-upload Qwen (state was wiped)
python3 -m scripts.upload --network local --canister llama_cpp \
  --canister-filename models/model.gguf \
  models/Qwen/qwen2.5-0.5b-instruct-q8_0.gguf

# Set max = 13 first (just to confirm onicai prefill needs ≥3 calls)
dfx canister call llama_cpp set_max_tokens \
  '(record { max_tokens_query = 1 : nat64; max_tokens_update = 13 : nat64 })'
dfx canister call llama_cpp load_model \
  '(record { args = vec {"--model"; "models/model.gguf"} })'
dfx canister call llama_cpp new_chat \
  '(record { args = vec {"--model"; "models/model.gguf"; "--prompt-cache"; "prompt.cache"} })'

# Prefill ×3 (onicai needs one more than Comoto)
for i in 1 2 3; do
  dfx canister call llama_cpp run_update \
    "(record { args = vec {\"--model\"; \"models/model.gguf\"; \"--prompt-cache\"; \"prompt.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-n\"; \"512\"; \"-p\"; \"$PROMPT\"} })"
done

# Now binary-search down to 10
dfx canister call llama_cpp set_max_tokens \
  '(record { max_tokens_query = 1 : nat64; max_tokens_update = 10 : nat64 })'

# Pure gen @ 10 → should return OK with output "Proof-of-work is used in blockchains where a"
dfx canister call llama_cpp run_update \
  "(record { args = vec {\"--model\"; \"models/model.gguf\"; \"--prompt-cache\"; \"prompt.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-n\"; \"512\"; \"-p\"; \"$PROMPT\"} })"

# Pure gen @ 11 → should TRAP
dfx canister call llama_cpp set_max_tokens \
  '(record { max_tokens_query = 1 : nat64; max_tokens_update = 11 : nat64 })'
dfx canister call llama_cpp run_update \
  "(record { args = vec {\"--model\"; \"models/model.gguf\"; \"--prompt-cache\"; \"prompt.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-n\"; \"512\"; \"-p\"; \"$PROMPT\"} })"
```

## 8. Tear down

```bash
dfx stop
```

## What you've established

| Metric                     | Expected | Your result |
| -------------------------- | -------- | ----------- |
| Comoto gen tok/call (OK)     | 29       | **\_**      |
| Comoto gen tok/call (TRAP)   | 30       | **\_**      |
| onicai gen tok/call (OK)   | 10       | **\_**      |
| onicai gen tok/call (TRAP) | 11       | **\_**      |
| Ratio                      | 2.9×     | **\_**      |

If your numbers match, the paper's central claim is reproduced for you.

## Why your WASM SHA256s will not byte-match the paper

Two reasons:

1. **LTO non-determinism.** Two consecutive builds of the same source on the same machine produce different binaries (~8% byte-divergence). `-flto` reorders symbols without `SOURCE_DATE_EPOCH` pinning.
2. **Toolchain drift.** clang/wasi-sdk/icpp-pro versions drift the binary even for identical source.

This affects the binary's hash but not its behavior — the instruction count for the hot inference path is unchanged, which is why the trap-boundary tok/call number is stable across rebuilds.

For full byte-exact reproducibility, a pinned Docker image with `SOURCE_DATE_EPOCH` is on the roadmap; it is not load-bearing for the paper claim.
