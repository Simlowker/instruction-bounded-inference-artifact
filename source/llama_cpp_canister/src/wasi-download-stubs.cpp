// WASI stubs for download.cpp and hf-cache.cpp
// These functions are called by arg.cpp/common.cpp but need cpp-httplib (no HTTP on WASI)

// Use relative path to avoid conflict with src/download.h (onicai)
#include "llama_cpp_onicai_fork/common/download.h"
#include "llama_cpp_onicai_fork/common/hf-cache.h"

std::pair<long, std::vector<char>> common_remote_get_content(const std::string &, const common_remote_params &) {
    return {-1, {}};
}

std::pair<std::string, std::string> common_download_split_repo_tag(const std::string & hf_repo_with_tag) {
    auto pos = hf_repo_with_tag.rfind(':');
    if (pos != std::string::npos && pos > 0) {
        return {hf_repo_with_tag.substr(0, pos), hf_repo_with_tag.substr(pos + 1)};
    }
    return {hf_repo_with_tag, ""};
}

std::vector<common_cached_model_info> common_list_cached_models() {
    return {};
}

common_download_model_result common_download_model(
    const common_params_model &,
    const std::string &,
    const common_download_model_opts &,
    const common_header_list &) {
    return {};
}

int common_download_file_single(const std::string &,
                                const std::string &,
                                const std::string &,
                                bool,
                                const common_header_list &,
                                bool) {
    return -1;
}

std::string common_docker_resolve_model(const std::string &) {
    return "";
}

namespace hf_cache {
void migrate_old_cache_to_hf_cache(const std::string &, bool) {}
}  // namespace hf_cache
