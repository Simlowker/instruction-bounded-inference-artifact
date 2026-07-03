// WASI exception ABI stubs — llama.cpp uses throw/catch but WASI
// doesn't have a real unwind runtime.  These stubs let the code
// compile & link; any actual throw will abort immediately.
//
// IMPORTANT: All code paths that may throw must be guarded with
// #ifdef __wasi__ to avoid reaching these stubs at runtime.
// Common patterns:
//   - warmup disabled in common.cpp (would throw during llama_decode)
//   - "--model is required" in arg.cpp returns false instead of throwing
//   - getenv() calls guarded to prevent garbage pointer dereference

#include <cstdlib>
#include <cstdio>
#include <cstring>

extern "C" {

void *__cxa_allocate_exception(unsigned long size) {
    // Allocate so __cxa_throw can inspect the exception data
    void *ptr = malloc(size);
    if (ptr) memset(ptr, 0, size);
    return ptr;
}

void __cxa_free_exception(void *ptr) {
    free(ptr);
}

__attribute__((noreturn))
void __cxa_throw(void *thrown_object, void *tinfo, void (*dest)(void *)) {
    fprintf(stderr, "FATAL: unhandled C++ exception on WASI\n");
    // Best-effort: try to print exception message by scanning object fields
    if (thrown_object) {
        const char **ptrs = (const char **)thrown_object;
        for (int i = 0; i < 8; i++) {
            const char *s = ptrs[i];
            if (s && (unsigned long)s > 0x1000 && (unsigned long)s < 0x7FFFFFFF) {
                bool ok = true;
                for (int j = 0; j < 128 && s[j]; j++) {
                    if (s[j] < 0x20 && s[j] != '\n' && s[j] != '\t') { ok = false; break; }
                }
                if (ok && s[0] != '\0') {
                    fprintf(stderr, "  exception[%d]: %.200s\n", i, s);
                }
            }
        }
    }
    abort();
}

void *__cxa_begin_catch(void *exc_obj) {
    return exc_obj;
}

void __cxa_end_catch() {}

}  // extern "C"
