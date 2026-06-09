# Execution-Boundary Evidence for Blockchain AI Claims

**Checked:** 2026-04-22

**Purpose.** This note supports the introduction and related-work framing in
`../drafts/paper-v14-short.md`. The key distinction is not "blockchain AI" in
the broad marketing sense, but **where the model forward pass executes**:
inside the consensus-metered smart-contract runtime, or outside it.

## Working Definitions

- **Native in-consensus inference:** the model forward pass executes inside the
  smart contract / canister runtime under the chain's replicated or
  consensus-metered compute budget.
- **Hybrid or off-chain-assisted inference:** the chain stores requests,
  verifies proofs, or orchestrates workers, but the forward pass itself occurs
  outside the replicated contract runtime.
- **Native deployment without throughput law:** the source demonstrates that
  on-chain / WASM inference is feasible, but does not derive a decoder
  throughput law such as `tok/call ≈ B / (alpha_eff x 2P)`.

## Verified Source Map

| Source | What the primary source establishes | Boundary classification | Relevance to this paper |
| --- | --- | --- | --- |
| ML2SC (`arXiv:2404.16967`) | PyTorch MLPs are translated to Solidity; training stays off-chain; inference runs by contract call; gas cost is modeled for these MLPs. | Native on-chain inference for small classical models. | Confirms that "native on-chain inference" is real, but not yet native LLM decoder execution. |
| On-Chain Decentralized Learning and Cost-Effective Inference for DeFi Attack Mitigation (`arXiv:2510.16024`) | Gas-prohibitive computation is moved to Layer 2; verified updates are propagated to Layer 1; bounded inference is done in smart contracts. | Hybrid training / bounded on-chain inference. | Supports the claim that some blockchain AI systems deliberately split the compute boundary. |
| Weaving the Cosmos (`arXiv:2502.17604`) | Proposes a WASM-based framework for AI inference across blockchain nodes and evaluates feasibility, scalability, and model security. | WASM-based blockchain AI framework, but systems-level. | Supports a feasibility claim for WASM-based blockchain AI outside ICP, but not a strong cross-chain deployment claim or a decoder throughput law. |
| `onicai/llama_cpp_canister` GitHub + ICP forum thread | `llama.cpp` is deployed as an Internet Computer smart contract / canister; the repo states that GGUF LLMs can run on-chain; the forum thread documents live use with DeepSeek 1.5B and Qwen 0.5B-class models. | Native in-canister LLM inference on ICP. | This is the immediate native-LLM baseline that our paper measures against and extends. |
| DFINITY LLM Canister forum announcement | Prompts are queued in a canister and processed by "AI workers"; the MVP uses workers outside the canister path, with future decentralization discussed separately. | Middleware / worker-based architecture. | Useful counterexample inside the ICP ecosystem itself: not every "ICP AI" system is native in-canister inference. |
| ICP docs: resource limits + deterministic SIMD | ICP documents a 40B instruction limit per update call and identifies executed Wasm instructions as an execution-throughput bottleneck. ICP also documents deterministic WebAssembly SIMD on every node. | Consensus-metered native Wasm runtime. | This is the execution model that makes the paper's cost law and measurements well-posed. |
| This paper's artifact (`data/onchain/*.csv`, `MAINNET-BENCHMARK-RESULTS.md`) | Local, ICP mainnet, and Swiss Subnet runs are recorded for the exact models discussed in the paper. | Native in-consensus measurements. | Turns the boundary discussion into measured evidence rather than architecture-only prose. |

## Primary Sources Used

- ML2SC: <https://arxiv.org/abs/2404.16967>
- DeFi attack mitigation paper: <https://arxiv.org/abs/2510.16024>
- Weaving the Cosmos: <https://arxiv.org/abs/2502.17604>
- onicai `llama_cpp_canister`: <https://github.com/onicai/llama_cpp_canister>
- ICP forum thread for `llama_cpp_canister`: <https://forum.dfinity.org/t/llama-cpp-on-the-internet-computer/33471>
- DFINITY LLM Canister architecture thread: <https://forum.dfinity.org/t/introducing-the-llm-canister-deploy-ai-agents-with-a-few-lines-of-code/41424>
- ICP resource limits: <https://docs.internetcomputer.org/building-apps/canister-management/resource-limits>
- ICP deterministic SIMD: <https://docs.internetcomputer.org/building-apps/network-features/simd>

## Realistic Native-LLM Comparison

The table below compares the three runtimes at the **contract/canister
boundary**. It does **not** compare custom validator binaries, sidecars,
workers, provers, or appchain-specific native modules, because those move the
forward pass outside the ordinary smart-contract execution path.

| Runtime | What the sources support | Practical LLM viability at the contract boundary | Main bottleneck | Evidence strength in this note | Realistic bottom line |
| --- | --- | --- | --- | --- | --- |
| EVM / Solidity | Native on-chain inference for small classical models is real (`ML2SC`), with gas modeled for MLP deployment, weight updates, and inference. | Very weak for decoder LLMs. The source base here supports small fixed-point MLPs, not public native LLM deployments. | Gas-metered stack-machine execution and per-transaction economics make dense decoder inference impractical long before useful LLM scales. | **Strong for small models; weak for native LLMs.** | Realistic for transparent toy models and narrow classifiers, not for useful native decoder LLM serving. |
| Cosmos / CosmWasm / WASM | WASM smart-contract execution with gas metering is real, and `Weaving the Cosmos` supports a feasibility-oriented framework for on-chain AI inference across nodes. | Plausible in principle, stronger than EVM as a compute substrate, but not demonstrated here with comparably strong public evidence for native decoder LLM deployments. | Gas-metered replicated Wasm execution plus chain-specific block-gas settings leave the feasible envelope highly chain-dependent. | **Medium for feasibility; weak for deployed native LLMs.** | Best framed as a promising WASM path, not as equally established public evidence of native on-chain LLM execution. |
| ICP / Canister Wasm | Native in-canister LLM inference is publicly documented, with deterministic Wasm SIMD, explicit instruction limits, and public `llama.cpp` canister deployments plus token-per-update data. | Strongest case in this source set for native sub-billion-parameter LLM inference inside the contract runtime itself. | Instruction-budget ceilings per update call dominate, so viability depends on quantization, kernel efficiency, and multi-call composition. | **Strong for native LLM execution.** | The only runtime in this comparison with clear public evidence of real native LLM execution at the smart-contract boundary. |

### Comparison Guardrail

If a Cosmos-style chain embeds a custom AI engine directly in validator
software, that may enable heavier inference than a normal smart contract.
However, that is a different comparison class than `EVM contract` vs
`CosmWasm contract` vs `ICP canister`, and should not be described as the same
kind of "100% on-chain smart-contract inference" without qualification.

## Safe Claims Supported by the Evidence

- The relevant blockchain-AI literature splits across **execution boundaries**,
  not just across chains.
- Some systems keep inference inside the contract/canister boundary; others use
  workers, Layer 2, proofs, or precomputed vectors.
- Outside ICP, the evidence here is strongest for small native on-chain models
  in EVM-style settings; WASM-based non-ICP systems are better framed as
  feasibility evidence unless accompanied by stronger deployment data.
- This paper studies the **native consensus-metered boundary** and measures it
  on ICP.
- ICP is the environment **in this study** where native sub-billion-parameter
  LLM inference is directly measured end-to-end.

## Claims Not Supported by the Evidence

- "ICP is the only blockchain where native on-chain inference is possible."
- "No other chain has native on-chain inference."
- "This paper proves a cross-chain impossibility result."

## Suggested Paper Framing

Use wording of the form:

> The broader question is blockchain AI, but the measurable question in this
> paper is what happens when the model forward pass itself runs inside a
> consensus-metered runtime.

Avoid wording of the form:

> We searched everywhere and discovered that only ICP can do this.

The first statement is supported by the evidence above. The second is not.

For a compact paper-facing version, use wording of the form:

> At the contract-runtime boundary, EVM evidence is strongest for small native
> on-chain models, Cosmos/WASM evidence is best treated as feasibility support,
> and ICP provides the clearest public evidence of native LLM execution.
