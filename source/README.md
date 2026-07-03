# source/ — Comoto fork source snapshot

`git archive` export of the subtree `llama_cpp_canister/` from the authors' private
monorepo at the paper's pinned commit (Table DA1: `Simlowker/gian@8cda13b`; repository
tag: `paper-v14-bench-snapshot`). Exported 2026-07-03. Five internal working notes
(`.claude/`, `docs/superpowers/` — planning documents, no build-relevant content) were
removed from the export; **all build sources are byte-identical to the pinned commit**.

- Builds the **29 tok/call** WASM of the 2.9× result (rebuild SHA-256 `da112d99…`, see
  `../results/rebench_2026-05-19.md`).
- The llama.cpp fork sources are **vendored** under
  `llama_cpp_canister/src/llama_cpp_onicai_fork/` (upstream `ggml-org/llama.cpp` lineage
  with canister/WASM-SIMD adaptations) — no extra clone needed.
- `common/build-info.cpp` is generated at build time — see `../REPRODUCE.md` §2.
- License: MIT (see `LICENSE`) — lineage: llama.cpp © ggml-org contributors; canister
  shell lineage onicai/llama_cpp_canister; canister/WASM-SIMD modifications © 2026 the
  paper authors.
