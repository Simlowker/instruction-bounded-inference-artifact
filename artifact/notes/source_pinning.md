# Source Pinning Note

This note records the source pins used by the v14 short manuscript and the
companion `instruction-bounded-inference-artifact` reproduction package. It is intentionally
separate from the CSV evidence tables: the tables record measured outputs,
while this note records source provenance and publication gates.

## Reproduction Artifact Pins

| Component | Pin | Evidence |
| --- | --- | --- |
| Comoto rebuild source | `Simlowker/gian@8cda13b` | Companion `instruction-bounded-inference-artifact` repository, files `REPRODUCE.md` and `results/rebench_2026-05-19.md` |
| onicai rebuild source | `onicai/llama_cpp_canister@v0.10.1` | Companion `instruction-bounded-inference-artifact` repository, files `REPRODUCE.md` and `results/rebench_2026-05-19.md` |
| Original onicai baseline tag | `onicai/llama_cpp_canister@v0.9.0` | `artifact/data/onchain/icp_mainnet.csv`; manuscript reference [9] |
| Qwen 2.5 0.5B Instruct Q8_0 GGUF | SHA-256 `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` | Companion `instruction-bounded-inference-artifact` repository, files `REPRODUCE.md` and `results/rebench_2026-05-19.md` |
| Rebuilt Comoto WASM | SHA-256 `da112d9916ea816620ddba02a09aead5c2fc966ac9026681880e71d388e9a49c` | Companion `instruction-bounded-inference-artifact` repository, file `results/rebench_2026-05-19.md` |
| Rebuilt onicai WASM | SHA-256 `b6ccbff0b6287026e7b91924d3bc77dbcfbfa35c17f3d0bd865a09d929169660` | Companion `instruction-bounded-inference-artifact` repository, file `results/rebench_2026-05-19.md` |

## Manuscript Reference Pins

| Reference | Source status | Publication action |
| --- | --- | --- |
| [6] DFINITY IC WASM instrumentation | The manuscript cites the public `dfinity/ic` path to `rs/embedders/src/wasm_utils/instrumentation.rs`, accessed April 2026. The exact April 2026 commit was not captured in the local package. | Before final archival deposit, replace the `master` URL in reference [6] with an immutable commit URL if the exact inspected commit is recoverable. If not recoverable, keep the access date and this caveat. |
| [10] llama.cpp | The paper's kernel comparison is anchored to the canister build and measured artifact rows. The exact upstream llama.cpp commit for every imported source file is not recoverable from the current local package. | Keep measured WASM/source pins above as the reproducibility anchor; add an immutable upstream commit only if it can be recovered from the build history. |

## Byte-Exact Rebuild Caveat

The original 2026-04-09 Comoto WASM hash (`ef8f9d78...`) is not byte-reproducible:
it was built from a working tree whose model source files were not all committed.
The publication claim is therefore functional reproducibility of the 29-vs-10
fork gap, not byte-exact reproduction of the original April binary.

## Registry GGUF provenance columns (`models.csv`)

`gguf_path` / `gguf_match` record the GGUF file used to **audit parameter counts**
(`params_M_gguf`), not necessarily the binary of the measured run (the `quant` column).
Parameter counts are quantization-invariant, so for four historical Q8_0 rows
(M01, M15, M18, M26) the Q4_0 GGUF of the same checkpoint was used for the parameter
audit; `gguf_match = exact` refers to the parameter-count match, not the measured quant.

## Upstream freshness checks

- **2026-07-03** — onicai `llama_cpp_canister` `v0.11.0` (released 2026-06-29) diff-checked
  against the pinned `v0.10.1` (GitHub compare, 2 commits, 11 files): adds an admin-gated
  cycle-balance monitor (`src/cycle_balance.*`, candid entry, tests) only. No llama.cpp,
  kernel, SIMD, or build-flag change — the 29-vs-10 baseline claim (ref [9]) is unaffected.
