# Metering-Overhead Multiplier `β = 1.844` — Derivation

This note documents how the metering-overhead multiplier `β` reported in
`drafts/paper-v14-short.md` §2 and §4.2 is derived. It also reconciles the
opcode-table cost ratio `f32 / i32 = 2×` with the observed SIMD-kernel-level
ratio `1.05×` recorded in `artifact/data/kernel/matmul_bench.csv`. Both
numbers are real, but they live at different levels of the cost stack and
must not be conflated.

## What `β` means

`β` is the per-iteration cost multiplier applied by the ICP metering
instrumentation pass [5, 6] on top of the underlying WebAssembly opcode
budget for a tight SIMD inner loop. We define

```
β = (instructions metered per iteration) / (instructions an unmetered
                                             tight loop would account for)
```

The metering pass injects a fixed cost-accounting block at re-entrant basic
blocks. On a tight SIMD inner loop that nominally executes ~4 opcodes, the
fixed instrumentation block inflates the per-iteration accounting, shifting
α_kernel toward `β × α_unmetered`.

## Source measurement

The derivation uses the empty-loop microbenchmark recorded in the constants
table inside `artifact/scripts/analyze_kernels.py` (the `METERING` dictionary
at the top of the file):

| Quantity | Value |
| --- | --- |
| `empty_loop_instr` | 177,460,428 ICP units |
| `empty_loop_iters` | 26,214,400 iterations |
| `cost_per_iter` (= empty_loop_instr / empty_loop_iters) | 6.770 ICP units / iteration |
| Unmetered baseline (tight ~4-opcode inner loop) | ~3.67 ICP units / iteration |
| `β` (= cost_per_iter / unmetered) | **1.844** |

The empty-loop measurement was recorded on the same canister build family
that produced `artifact/data/kernel/matmul_bench.csv` (`fork build
17806d52`). The measurement is preserved in the script's constants table so
that re-running `python3 artifact/scripts/analyze_kernels.py` produces a
self-consistent derivation; the underlying observation does not currently
have its own row in `matmul_bench.csv`. Future re-collections should add the
empty-loop row to the CSV directly so the derivation rebuilds end-to-end
from data files.

## Reconciling `2×` opcode cost with `1.05×` observed SIMD-kernel ratio

The paper's §1.2 regime table lists `f32.mul vs i32.mul = 2×` from the ICP
opcode cost table [7]. The CSV row at line 13–14 of `matmul_bench.csv`
reports the observed SIMD-kernel-level ratio:

```
SIMD_f32x4_mul_add ... 103.2 M instructions
SIMD_i32x4_mul_add ...  98.3 M instructions
ratio = 1.05× (vs 2× expected)
```

Both numbers are correct; they describe two different things:

1. **Opcode-table cost (2×)** is the per-opcode price of `f32.mul` (2 ICP
   units) versus `i32.mul` (1 ICP unit) under the ICP cost table [7]. This
   is a static schedule fact about the runtime, independent of the loop body
   that surrounds the multiply.

2. **Observed SIMD-kernel ratio (1.05×)** is the end-of-loop accounted cost
   for a tight SIMD `f32x4_mul_add` inner loop versus its `i32x4_mul_add`
   analogue, measured over `100 × 512 × 512` operations at SIMD width 4.
   At that level the metering instrumentation block dominates the per-
   iteration accounting (`β = 1.844`), so the underlying `f32`/`i32` cost
   asymmetry is largely absorbed by the fixed-cost prologue/epilogue of the
   instrumentation pass. The remaining `~5%` gap is the residual visible
   asymmetry after `β` washout.

The practical consequence (paper §4.2): kernel-table opcode differences
between `f32` and `i32` are drowned by metering overhead at the SIMD-kernel
level, which is why the only end-to-end throughput lever is effective matmul
cost, not opcode-mix engineering.

## Where the numbers live in the artifact

- Empty-loop derivation constants: `artifact/scripts/analyze_kernels.py`
  (`METERING` dict, near top of the file)
- `f32x4` vs `i32x4` SIMD-kernel ratio: `artifact/data/kernel/matmul_bench.csv`
  rows `SIMD_f32x4_mul_add` and `SIMD_i32x4_mul_add` (`f32x4/i32x4 ratio = 1.05x`)
- ICP per-opcode cost table: linked from reference [7] in the manuscript

## Caveats

- The empty-loop microbench has not been re-collected on the April 2026
  build line; build sensitivity for `β` itself has not been swept.
- The 1.05× kernel-level ratio is specific to the tight-loop SIMD form
  measured here; loop bodies with non-trivial scheduling pressure or with
  per-iteration branching would re-introduce more of the `2×` opcode-table
  asymmetry.
- The `β = 1.844` value should be treated as a current-build operating
  point, not as a runtime-wide constant of nature.
