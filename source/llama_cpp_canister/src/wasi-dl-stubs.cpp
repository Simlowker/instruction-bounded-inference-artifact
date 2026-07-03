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
