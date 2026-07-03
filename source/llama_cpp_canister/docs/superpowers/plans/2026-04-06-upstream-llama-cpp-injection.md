# Upstream llama.cpp Injection into onicai Canister — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old onicai llama.cpp fork with upstream llama.cpp inside the existing canister project, preserving the canister wrapper and ICP deployment infrastructure.

**Architecture:** The onicai project (`llama_cpp_canister/`) provides the ICP canister shell (auth, upload, files, inference endpoints). We swap `src/llama_cpp_onicai_fork/` with upstream llama.cpp, add WASI shims from `llama-icp/`, apply 5 targeted `#ifdef __wasi__` patches, and write a new `icpp.toml` that combines the llama-icp build config with the onicai canister wrapper.

**Tech Stack:** C/C++ (WASM target), icpp-pro toolchain, wasi-sdk-25.0, dfx (ICP local network), WASM SIMD128

**Key reference:** The `llama-icp` project (`/Users/macpromax/gian/llama-icp/`) already solved the upstream compilation. Its `icpp.toml`, WASI patches, and shims are the primary source of truth.

---

### Task 1: Backup old fork and prepare directory structure

**Files:**
- Rename: `src/llama_cpp_onicai_fork/` → `src/llama_cpp_onicai_fork_backup/`
- Create: `src/llama_cpp_onicai_fork/` (will contain upstream)
- Create: `wasi-shims/` (top-level, alongside src/)

- [ ] **Step 1: Create a git branch for this work**

```bash
cd /Users/macpromax/gian/llama_cpp_canister
git checkout -b feat/upstream-llama-cpp
```

- [ ] **Step 2: Backup the old fork**

```bash
mv src/llama_cpp_onicai_fork src/llama_cpp_onicai_fork_backup
```

- [ ] **Step 3: Verify backup**

```bash
ls src/llama_cpp_onicai_fork_backup/src/llama.cpp
```
Expected: file exists

- [ ] **Step 4: Commit the backup**

```bash
git add -A
git commit -m "chore: backup old onicai fork before upstream injection"
```

---

### Task 2: Copy upstream llama.cpp into the project

**Files:**
- Create: `src/llama_cpp_onicai_fork/` (populated from upstream)

The upstream clone is at `/Users/macpromax/gian/llama-cpp-upstream/`. We only need specific subdirectories — NOT the full repo (no examples, tests, cmake, scripts, etc.).

- [ ] **Step 1: Create the directory structure**

```bash
cd /Users/macpromax/gian/llama_cpp_canister
mkdir -p src/llama_cpp_onicai_fork
```

- [ ] **Step 2: Copy only the needed directories from upstream**

```bash
UPSTREAM=/Users/macpromax/gian/llama-cpp-upstream
DEST=src/llama_cpp_onicai_fork

# Core source
cp -R "$UPSTREAM/src" "$DEST/src"
cp -R "$UPSTREAM/include" "$DEST/include"
cp -R "$UPSTREAM/ggml" "$DEST/ggml"
cp -R "$UPSTREAM/common" "$DEST/common"

# Vendor (JSON header-only lib)
mkdir -p "$DEST/vendor"
cp -R "$UPSTREAM/vendor/nlohmann" "$DEST/vendor/nlohmann"
```

- [ ] **Step 3: Verify key files exist**

```bash
# Core
ls src/llama_cpp_onicai_fork/src/llama.cpp
# New graph/io files
ls src/llama_cpp_onicai_fork/src/llama-graph.cpp
ls src/llama_cpp_onicai_fork/src/llama-io.cpp
# Memory management
ls src/llama_cpp_onicai_fork/src/llama-memory.cpp
# Model architectures
ls src/llama_cpp_onicai_fork/src/models/llama.cpp
ls src/llama_cpp_onicai_fork/src/models/qwen2.cpp
# WASM SIMD kernels (THE critical file)
ls src/llama_cpp_onicai_fork/ggml/src/ggml-cpu/arch/wasm/quants.c
# New ggml-cpu ops
ls src/llama_cpp_onicai_fork/ggml/src/ggml-cpu/ops.cpp
ls src/llama_cpp_onicai_fork/ggml/src/ggml-cpu/vec.cpp
# Vendor JSON
ls src/llama_cpp_onicai_fork/vendor/nlohmann/json.hpp
```
Expected: all files exist

- [ ] **Step 4: Commit**

```bash
git add src/llama_cpp_onicai_fork
git commit -m "feat: add upstream llama.cpp source tree"
```

---

### Task 3: Copy WASI shims and stubs

**Files:**
- Create: `wasi-shims/` (from `/Users/macpromax/gian/llama-icp/wasi-shims/`)
- Create: `src/wasi-exception-stubs.cpp`
- Create: `src/wasi-dl-stubs.cpp`

The wasi-shims provide no-op threading primitives (`<mutex>`, `<thread>`, `<condition_variable>`, `<future>`) for ICP's single-threaded environment. The stubs handle exception ABI and dynamic loading.

- [ ] **Step 1: Copy wasi-shims directory**

```bash
cd /Users/macpromax/gian/llama_cpp_canister
cp -R /Users/macpromax/gian/llama-icp/wasi-shims .
```

- [ ] **Step 2: Verify shim files**

```bash
ls wasi-shims/mutex wasi-shims/thread wasi-shims/condition_variable wasi-shims/future
ls wasi-shims/__mutex/once_flag.h
```
Expected: all files exist

- [ ] **Step 3: Create wasi-exception-stubs.cpp**

Write `src/wasi-exception-stubs.cpp`:
```cpp
// WASI exception ABI stubs — llama.cpp uses throw/catch but WASI
// doesn't have a real unwind runtime.  These stubs let the code
// compile & link; any actual throw will abort immediately.

#include <cstdlib>
#include <cstdio>

extern "C" {

void *__cxa_allocate_exception(unsigned long) {
    fprintf(stderr, "FATAL: exception allocation on WASI (unsupported)\n");
    abort();
    return nullptr; // unreachable
}

void __cxa_throw(void *, void *, void (*)(void *)) {
    fprintf(stderr, "FATAL: exception throw on WASI (unsupported)\n");
    abort();
}

void *__cxa_begin_catch(void *) {
    return nullptr;
}

void __cxa_end_catch() {}

}  // extern "C"
```

- [ ] **Step 4: Create wasi-dl-stubs.cpp**

Write `src/wasi-dl-stubs.cpp`:
```cpp
// WASI stubs for dynamic loading — no dlopen on ICP canisters
#include <cstdio>

extern "C" {

int dlclose(void *) { return 0; }

char *dlerror(void) {
    static char msg[] = "dlopen not supported on WASI";
    return msg;
}

void *dlopen(const char *, int) { return nullptr; }

void *dlsym(void *, const char *) { return nullptr; }

}  // extern "C"
```

- [ ] **Step 5: Commit**

```bash
git add wasi-shims src/wasi-exception-stubs.cpp src/wasi-dl-stubs.cpp
git commit -m "feat: add WASI shims and stubs for ICP single-threaded env"
```

---

### Task 4: Apply WASI patches to upstream files

**Files:**
- Modify: `src/llama_cpp_onicai_fork/ggml/src/ggml.c`
- Modify: `src/llama_cpp_onicai_fork/common/common.cpp`
- Modify: `src/llama_cpp_onicai_fork/common/log.cpp`
- Modify: `src/llama_cpp_onicai_fork/common/arg.cpp`
- Modify: `src/llama_cpp_onicai_fork/src/llama-sampler.cpp`

All patches use `#ifdef __wasi__` guards — they don't break non-WASI compilation.

**Reference:** Exact patches documented at `/Users/macpromax/gian/llama-icp/src/llama.cpp/` (these files ARE the patched versions).

- [ ] **Step 1: Patch ggml.c — timing functions**

Find the platform-specific `ggml_time_init`/`ggml_time_ms`/`ggml_time_us` block (look for `#elif defined(__APPLE__)` or the `clock_gettime` calls). Add a `__wasi__` block BEFORE the other platform checks:

```c
#if defined(__wasi__)
/* WASI/ICP: no clock_gettime, return 0 (timers not needed for inference) */
void ggml_time_init(void) {}
int64_t ggml_time_ms(void) { return 0; }
int64_t ggml_time_us(void) { return 0; }
#elif defined(__APPLE__) && defined(__MACH__)
// ... existing Apple code ...
```

- [ ] **Step 2: Patch common.cpp — 4 changes**

**2a. Conditional chrono include** — Find `#include <chrono>` near the top, wrap it:
```cpp
#ifndef __wasi__
#include <chrono>
#endif
```

**2b. set_process_priority()** — Find the function `set_process_priority`. Add `__wasi__` block:
```cpp
#elif defined(__wasi__)
bool set_process_priority(enum ggml_sched_priority prio) {
    (void)prio;
    return true; // no-op on WASI
}
```

**2c. string_get_sortable_timestamp()** — Find this function. Add at the start of its body:
```cpp
#ifdef __wasi__
    return "2026_01_01-00_00_00_000000000"; // WASI/ICP: no clock
#else
    // ... existing system_clock code ...
#endif
```

**2d. get_cache_directory()** — Find this function. Add `__wasi__` to the Emscripten guard:
```cpp
#elif defined(__EMSCRIPTEN__) || defined(__wasi__)
        GGML_ABORT("not implemented on this platform");
```

- [ ] **Step 3: Patch log.cpp — 2 changes**

**3a. Conditional chrono include** — Find `#include <chrono>`, wrap it:
```cpp
#ifndef __wasi__
#include <chrono>
#endif
```

**3b. t_us() function** — Find `static int64_t t_us()`. Replace or guard:
```cpp
static int64_t t_us() {
#ifdef __wasi__
    return 0; // WASI/ICP: no clock
#else
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
#endif
}
```

- [ ] **Step 4: Patch arg.cpp — syslimits guard**

Find the platform-specific `#include <linux/limits.h>` / `#include <sys/syslimits.h>` block. Wrap the entire block:
```cpp
#if !defined(__EMSCRIPTEN__) && !defined(__wasi__)
#ifdef __linux__
#include <linux/limits.h>
#elif defined(_WIN32)
// ... existing Windows code ...
#else
#include <sys/syslimits.h>
#endif
#endif
```

- [ ] **Step 5: Patch llama-sampler.cpp — 2 changes**

**5a. Conditional chrono include** — Find `#include <chrono>`, wrap it:
```cpp
#ifndef __wasi__
#include <chrono>
#endif
```

**5b. get_rng_seed()** — Find the `get_rng_seed` function. In the fallback branch that uses `system_clock`, add:
```cpp
#ifdef __wasi__
            return 42; // WASI/ICP: no clock, fixed seed
#else
            return (uint32_t) std::chrono::system_clock::now().time_since_epoch().count();
#endif
```

- [ ] **Step 6: Verify patches by diffing against reference**

```bash
# Compare each patched file against the llama-icp reference to ensure correctness
diff <(grep -n "__wasi__" src/llama_cpp_onicai_fork/ggml/src/ggml.c) <(grep -n "__wasi__" /Users/macpromax/gian/llama-icp/src/llama.cpp/ggml/src/ggml.c)
```
Expected: same `__wasi__` guards in both files

- [ ] **Step 7: Commit**

```bash
git add src/llama_cpp_onicai_fork/ggml/src/ggml.c \
        src/llama_cpp_onicai_fork/common/common.cpp \
        src/llama_cpp_onicai_fork/common/log.cpp \
        src/llama_cpp_onicai_fork/common/arg.cpp \
        src/llama_cpp_onicai_fork/src/llama-sampler.cpp
git commit -m "feat: apply WASI patches to upstream llama.cpp files"
```

---

### Task 5: Create build-info.cpp stub

**Files:**
- Create: `src/llama_cpp_onicai_fork/common/build-info.cpp`

Upstream generates this from `build-info.cpp.in` during cmake build. We provide a static stub.

- [ ] **Step 1: Write build-info.cpp**

```cpp
// Static build info for ICP WASM build
int LLAMA_BUILD_NUMBER = 0;
char const *LLAMA_COMMIT = "upstream";
char const *LLAMA_COMPILER = "wasi-sdk clang";
char const *LLAMA_BUILD_TARGET = "wasm32-wasi";
```

- [ ] **Step 2: Commit**

```bash
git add src/llama_cpp_onicai_fork/common/build-info.cpp
git commit -m "feat: add static build-info.cpp for WASM target"
```

---

### Task 6: Fix canister wrapper compatibility

**Files:**
- Modify: `src/main_.cpp`

**Problem:** `main_.cpp` includes `"chat-template.hpp"` which existed in the old fork but NOT in upstream. Upstream moved chat template functions to `common/chat.h` with a different API:
- Old: `common_chat_templates_from_model()` → New: `common_chat_templates_init()` (returns `common_chat_templates_ptr`)
- Old: `chat_templates.has_explicit_template` → New: `common_chat_templates_was_explicit(tmpls.get())`
- Old: `*chat_templates.template_default` → New: `tmpls.get()`

Including `common/chat.cpp` would pull in 14+ additional files (jinja/, peg-parser, regex-partial, etc.). **Strategy: disable chat template code for first build, re-enable later.**

- [ ] **Step 1: Read main_.cpp fully to identify all chat template references**

```bash
grep -n "chat.template\|chat_template\|ChatTemplate\|chat-template" src/main_.cpp
```

- [ ] **Step 2: Guard chat template code with a compile-time flag**

Add at the top of `main_.cpp` (after the ICPP-PATCH block):
```cpp
// Temporarily disable chat templates — upstream requires common/chat.cpp + jinja/ (14+ files)
// Re-enable after core inference is validated
#define LLAMA_ICP_NO_CHAT_TEMPLATES
```

Replace `#include "chat-template.hpp"` with:
```cpp
#ifndef LLAMA_ICP_NO_CHAT_TEMPLATES
#include "chat.h"
#endif
```

Wrap all `chat_templates` usage blocks with `#ifndef LLAMA_ICP_NO_CHAT_TEMPLATES` / `#endif`.

Key blocks to wrap (approximate lines — verify by reading the file):
- Lines ~222-225: `auto chat_templates = common_chat_templates_from_model(...)`
- Lines ~284-313: `has_chat_template` checks and related LOG calls
- Lines ~383-397: `format_chat` lambda and `chat_add_and_format`
- Lines ~644: conversation mode chat template check
- Lines ~1105-1172: chat template at end-of-generation

- [ ] **Step 3: Check for other missing headers**

`main_.cpp` also includes `"console.h"`. Verify if console functions are used in ICPP-patched code. If not, guard the include:
```cpp
#ifndef __wasi__
#include "console.h"
#endif
```

- [ ] **Step 4: Check common_sampler API compatibility**

The old fork used `common_sampler` — verify the API hasn't changed in upstream. Key functions:
- `common_sampler_init()`
- `common_sampler_sample()`
- `common_sampler_accept()`
- `common_sampler_reset()`
- `common_sampler_free()`

```bash
grep -n "common_sampler" /Users/macpromax/gian/llama-cpp-upstream/common/sampling.h | head -20
```

- [ ] **Step 5: Compile check — look for other API breaks**

This step is iterative. After the first `icpp build-wasm`, fix any remaining API incompatibilities in the canister wrapper files.

- [ ] **Step 6: Commit**

```bash
git add src/main_.cpp
git commit -m "feat: guard chat template code for upstream compatibility"
```

---

### Task 7: Write new icpp.toml

**Files:**
- Modify: `icpp.toml`

This is the most critical file. Based on the working `llama-icp/icpp.toml` but adapted for onicai paths and canister wrapper.

- [ ] **Step 1: Back up current icpp.toml**

```bash
cp icpp.toml icpp.toml.backup
```

- [ ] **Step 2: Write the new icpp.toml**

```toml
# Upstream llama.cpp for ICP — injected into onicai canister shell
# Strategy: wasi-shims provide no-op threading, -fexceptions avoids throw patching

[build-wasm]
canister = "llama_cpp"
did_path = "src/llama_cpp.did"

cpp_paths = [
    # --- llama.cpp core (27 files) ---
    "src/llama_cpp_onicai_fork/src/llama.cpp",
    "src/llama_cpp_onicai_fork/src/llama-adapter.cpp",
    "src/llama_cpp_onicai_fork/src/llama-arch.cpp",
    "src/llama_cpp_onicai_fork/src/llama-batch.cpp",
    "src/llama_cpp_onicai_fork/src/llama-chat.cpp",
    "src/llama_cpp_onicai_fork/src/llama-context.cpp",
    "src/llama_cpp_onicai_fork/src/llama-cparams.cpp",
    "src/llama_cpp_onicai_fork/src/llama-grammar.cpp",
    "src/llama_cpp_onicai_fork/src/llama-graph.cpp",
    "src/llama_cpp_onicai_fork/src/llama-hparams.cpp",
    "src/llama_cpp_onicai_fork/src/llama-impl.cpp",
    "src/llama_cpp_onicai_fork/src/llama-io.cpp",
    "src/llama_cpp_onicai_fork/src/llama-kv-cache.cpp",
    "src/llama_cpp_onicai_fork/src/llama-kv-cache-iswa.cpp",
    "src/llama_cpp_onicai_fork/src/llama-memory.cpp",
    "src/llama_cpp_onicai_fork/src/llama-memory-hybrid.cpp",
    "src/llama_cpp_onicai_fork/src/llama-memory-hybrid-iswa.cpp",
    "src/llama_cpp_onicai_fork/src/llama-memory-recurrent.cpp",
    "src/llama_cpp_onicai_fork/src/llama-mmap.cpp",
    "src/llama_cpp_onicai_fork/src/llama-model.cpp",
    "src/llama_cpp_onicai_fork/src/llama-model-loader.cpp",
    "src/llama_cpp_onicai_fork/src/llama-quant.cpp",
    "src/llama_cpp_onicai_fork/src/llama-sampler.cpp",
    "src/llama_cpp_onicai_fork/src/llama-vocab.cpp",
    "src/llama_cpp_onicai_fork/src/unicode.cpp",
    "src/llama_cpp_onicai_fork/src/unicode-data.cpp",

    # --- model architectures (wildcard — 113 files, all arches) ---
    "src/llama_cpp_onicai_fork/src/models/*.cpp",

    # --- common (minimal set for inference) ---
    "src/llama_cpp_onicai_fork/common/arg.cpp",
    "src/llama_cpp_onicai_fork/common/common.cpp",
    "src/llama_cpp_onicai_fork/common/json-schema-to-grammar.cpp",
    "src/llama_cpp_onicai_fork/common/log.cpp",
    "src/llama_cpp_onicai_fork/common/sampling.cpp",
    "src/llama_cpp_onicai_fork/common/build-info.cpp",

    # --- ggml C++ ---
    "src/llama_cpp_onicai_fork/ggml/src/ggml.cpp",
    "src/llama_cpp_onicai_fork/ggml/src/gguf.cpp",
    "src/llama_cpp_onicai_fork/ggml/src/ggml-backend.cpp",
    "src/llama_cpp_onicai_fork/ggml/src/ggml-backend-reg.cpp",
    "src/llama_cpp_onicai_fork/ggml/src/ggml-opt.cpp",
    "src/llama_cpp_onicai_fork/ggml/src/ggml-threading.cpp",

    # --- ggml-cpu C++ ---
    "src/llama_cpp_onicai_fork/ggml/src/ggml-cpu/ggml-cpu.cpp",
    "src/llama_cpp_onicai_fork/ggml/src/ggml-cpu/ops.cpp",
    "src/llama_cpp_onicai_fork/ggml/src/ggml-cpu/binary-ops.cpp",
    "src/llama_cpp_onicai_fork/ggml/src/ggml-cpu/unary-ops.cpp",
    "src/llama_cpp_onicai_fork/ggml/src/ggml-cpu/vec.cpp",
    "src/llama_cpp_onicai_fork/ggml/src/ggml-cpu/traits.cpp",
    "src/llama_cpp_onicai_fork/ggml/src/ggml-cpu/repack.cpp",

    # --- WASI stubs ---
    "src/wasi-exception-stubs.cpp",
    "src/wasi-dl-stubs.cpp",

    # --- canister wrapper (onicai) ---
    "src/*.cpp",
]

# IMPORTANT: wasi-shims FIRST — overrides system <mutex>, <thread>, etc.
cpp_include_dirs = [
    "wasi-shims",
    "src/llama_cpp_onicai_fork/vendor",
    "src/llama_cpp_onicai_fork",
    "src/llama_cpp_onicai_fork/include",
    "src/llama_cpp_onicai_fork/src",
    "src/llama_cpp_onicai_fork/ggml/include",
    "src/llama_cpp_onicai_fork/ggml/src",
    "src/llama_cpp_onicai_fork/ggml/src/ggml-cpu",
    "src/llama_cpp_onicai_fork/common",
]

cpp_compile_flags = [
    "-DNDEBUG",
    "-DGGML_USE_CPU",
    "-fexceptions",
    "-D_WASI_EMULATED_SIGNAL",
    "-D_WASI_EMULATED_PTHREAD",
    "-D_WASI_EMULATED_GETPID",
    "-D_WASI_EMULATED_PROCESS_CLOCKS",
    "-DGGML_DEFAULT_N_THREADS=1",
]

# 8MB stack (sufficient for Qwen2.5, increase for larger models)
cpp_link_flags = [
    "-Wl,-z,stack-size=8388608",
    "-lwasi-emulated-pthread",
    "-lwasi-emulated-getpid",
    "-lwasi-emulated-signal",
    "-lwasi-emulated-process-clocks",
    "-fexceptions",
]

c_paths = [
    "src/llama_cpp_onicai_fork/ggml/src/ggml.c",
    "src/llama_cpp_onicai_fork/ggml/src/ggml-alloc.c",
    "src/llama_cpp_onicai_fork/ggml/src/ggml-quants.c",
    "src/llama_cpp_onicai_fork/ggml/src/ggml-cpu/ggml-cpu.c",
    "src/llama_cpp_onicai_fork/ggml/src/ggml-cpu/quants.c",
    # THE critical WASM SIMD file — 9 kernels
    "src/llama_cpp_onicai_fork/ggml/src/ggml-cpu/arch/wasm/quants.c",
]

c_include_dirs = [
    "src/llama_cpp_onicai_fork",
    "src/llama_cpp_onicai_fork/include",
    "src/llama_cpp_onicai_fork/ggml/include",
    "src/llama_cpp_onicai_fork/ggml/src",
    "src/llama_cpp_onicai_fork/ggml/src/ggml-cpu",
    "src/llama_cpp_onicai_fork/common",
]

c_compile_flags = [
    "-DNDEBUG",
    "-msimd128",
    "-DGGML_USE_CPU",
    "-D_WASI_EMULATED_SIGNAL",
    "-D_WASI_EMULATED_PTHREAD",
    "-D_WASI_EMULATED_GETPID",
    "-D_WASI_EMULATED_PROCESS_CLOCKS",
]

post_wasm_function = "scripts.optimize_wasm.main"

[build-native]
cpp_paths = [
    "native/*.cpp",
]
cpp_include_dirs = [
    "src/llama_cpp_onicai_fork",
    "src/llama_cpp_onicai_fork/include",
    "src/llama_cpp_onicai_fork/src",
    "src/llama_cpp_onicai_fork/ggml/include",
    "src/llama_cpp_onicai_fork/ggml/src",
    "src/llama_cpp_onicai_fork/common",
]
cpp_compile_flags = ["-DNDEBUG", "-DGGML_USE_CPU"]
cpp_link_flags = []
c_paths = []
c_include_dirs = []
c_compile_flags = ["-DNDEBUG", "-DGGML_USE_CPU"]
```

**Key differences from old icpp.toml:**
- `wasi-shims` as FIRST include dir (overrides system `<mutex>`, `<thread>`, etc.)
- `vendor/` include dir (for `nlohmann/json.hpp`)
- `-fexceptions` in both compile and link flags
- WASI emulation defines (`-D_WASI_EMULATED_*`)
- WASI emulation link libs (`-lwasi-emulated-*`)
- `-DGGML_DEFAULT_N_THREADS=1` (force single-thread)
- New files: `ggml.cpp`, `ggml-opt.cpp`, `ops.cpp`, `binary-ops.cpp`, `unary-ops.cpp`, `vec.cpp`, `repack.cpp`
- New files: `llama-graph.cpp`, `llama-io.cpp`, `llama-cparams.cpp`, `llama-memory*.cpp`, `llama-kv-cache-iswa.cpp`
- Model architectures: `src/models/*.cpp` (wildcard)
- WASM SIMD: `arch/wasm/quants.c` (9 kernels — was NOT in old icpp.toml!)
- Stubs: `wasi-exception-stubs.cpp`, `wasi-dl-stubs.cpp`

- [ ] **Step 3: Commit**

```bash
git add icpp.toml icpp.toml.backup
git commit -m "feat: new icpp.toml for upstream llama.cpp with WASI shims"
```

---

### Task 8: First build attempt

**Files:** None (build step)

- [ ] **Step 1: Activate conda environment**

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate llama_cpp_canister
```

If the conda env doesn't exist, try:
```bash
conda activate onicai
```
Or just run without conda if icpp is in PATH.

- [ ] **Step 2: Generate build-info if Makefile target exists**

```bash
cd /Users/macpromax/gian/llama_cpp_canister
make build-info-cpp-wasm 2>/dev/null || echo "Skipping — using static build-info.cpp"
```

- [ ] **Step 3: Run icpp build-wasm**

```bash
icpp build-wasm 2>&1 | head -200
```

Expected outcomes:
- **Success:** WASM file generated in `build/`
- **Likely errors:** Missing includes, API changes in canister wrapper, missing source files

- [ ] **Step 4: Fix errors iteratively**

Common error patterns and fixes:

| Error | Fix |
|-------|-----|
| `fatal error: 'chat-template.hpp' not found` | Verify Task 6 guard was applied |
| `fatal error: 'console.h' not found` | Guard with `#ifndef __wasi__` or add to common/ includes |
| `undefined reference to common_chat_*` | Chat template functions still referenced — guard more code |
| `undefined reference to __cxa_*` | Stubs not linked — check stubs are in cpp_paths |
| `fatal error: 'nlohmann/json.hpp' not found` | Add `vendor/` to include path, or `vendor/nlohmann/` |
| `'chrono' file not found` (in other files) | Apply same `#ifndef __wasi__` guard pattern |
| `redefinition of '__cxa_*'` | icpp-pro already provides these — remove our stubs |
| `multiple definition of 'dlopen'` | icpp-pro already provides these — remove our dl stubs |
| `undefined symbol: clock_gettime` | ggml.c patch not applied correctly |
| `undefined reference to ggml_opt_*` | ggml-opt.cpp missing from cpp_paths |
| `model arch not found` | models/*.cpp wildcard not working — list files individually |

- [ ] **Step 5: After successful build, check WASM size**

```bash
ls -lh build/llama_cpp.wasm
```

Expected: 2-10 MB (larger than old fork due to 113 model files)

- [ ] **Step 6: Commit fixes**

```bash
git add -A
git commit -m "fix: resolve build errors for upstream llama.cpp WASM"
```

---

### Task 9: Deploy locally and benchmark

**Files:** None (deployment step)

- [ ] **Step 1: Start local dfx network**

```bash
dfx start --clean --background
```

- [ ] **Step 2: Deploy canister**

```bash
dfx deploy --network local
```

- [ ] **Step 3: Upload SmolLM2-135M Q4_0 model (87 MB)**

```bash
# Use existing upload script
python scripts/upload.py --network local --model models/SmolLM2-135M-Instruct-Q4_0.gguf
```

If the model file isn't in models/, download it first:
```bash
# Check if model exists
ls models/*SmolLM2*Q4_0* 2>/dev/null
```

- [ ] **Step 4: Load model in canister**

```bash
dfx canister call llama_cpp load_model '(record { model = "/models/SmolLM2-135M-Instruct-Q4_0.gguf" })' --network local
```

- [ ] **Step 5: Benchmark SmolLM2 — measure tok/call**

```bash
# Single inference call
dfx canister call llama_cpp run_update '(record { prompt = "Hello"; max_tokens = 10 })' --network local
```

Measure tokens generated per call. **Target: >= 103 tok/call** (onicai baseline).

- [ ] **Step 6: Upload Qwen 2.5 0.5B Q4_0 (336 MB) and benchmark**

```bash
python scripts/upload.py --network local --model models/qwen2.5-0.5b-instruct-q4_0.gguf
dfx canister call llama_cpp load_model '(record { model = "/models/qwen2.5-0.5b-instruct-q4_0.gguf" })' --network local
dfx canister call llama_cpp run_update '(record { prompt = "Hello"; max_tokens = 10 })' --network local
```

**Target: >= 27 tok/call** (onicai baseline).

- [ ] **Step 7: Record results**

Create a benchmark results file:
```
SmolLM2-135M Q4_0:
- tok/call upstream: ???
- tok/call onicai:   103
- delta: ???

Qwen 2.5 0.5B Q4_0:
- tok/call upstream: ???
- tok/call onicai:   27
- delta: ???

WASM binary size: ??? MB (old: ~848 KB)
9 SIMD kernels active: yes/no
```

- [ ] **Step 8: Commit and tag**

```bash
git add -A
git commit -m "feat: upstream llama.cpp injection — benchmark results"
```

---

### Task 10 (future): Re-enable chat templates

**Scope:** After core inference is validated, add chat template support.

**Files to add to icpp.toml cpp_paths:**
```
"src/llama_cpp_onicai_fork/common/chat.cpp",
"src/llama_cpp_onicai_fork/common/chat-auto-parser-generator.cpp",
"src/llama_cpp_onicai_fork/common/chat-auto-parser-helpers.cpp",
"src/llama_cpp_onicai_fork/common/chat-peg-parser.cpp",
"src/llama_cpp_onicai_fork/common/peg-parser.cpp",
"src/llama_cpp_onicai_fork/common/regex-partial.cpp",
"src/llama_cpp_onicai_fork/common/json-partial.cpp",
"src/llama_cpp_onicai_fork/common/jinja/caps.cpp",
"src/llama_cpp_onicai_fork/common/jinja/lexer.cpp",
"src/llama_cpp_onicai_fork/common/jinja/parser.cpp",
"src/llama_cpp_onicai_fork/common/jinja/runtime.cpp",
"src/llama_cpp_onicai_fork/common/jinja/string.cpp",
"src/llama_cpp_onicai_fork/common/jinja/value.cpp",
```

**API migration in main_.cpp:**
- `#include "chat-template.hpp"` → `#include "chat.h"`
- `common_chat_templates_from_model(model, params.chat_template)` → `common_chat_templates_init(model, params.chat_template.c_str(), nullptr)`
- `chat_templates.has_explicit_template` → `common_chat_templates_was_explicit(tmpls.get())`
- `*chat_templates.template_default` → `tmpls.get()`
- `common_chat_format_single(*chat_templates.template_default, ...)` → `common_chat_format_single(tmpls.get(), ...)`
- `common_chat_format_example(*chat_templates.template_default, ...)` → `common_chat_format_example(tmpls.get(), ...)`

---

### Task 11 (future): Add TQ2_0 SIMD kernel

**Scope:** After upstream benchmarks are validated, port the custom TQ2_0 SIMD kernel (paper's original contribution) into the upstream WASM quants file.

**Target file:** `src/llama_cpp_onicai_fork/ggml/src/ggml-cpu/arch/wasm/quants.c`

This is the step that produces novel research value for the paper.
