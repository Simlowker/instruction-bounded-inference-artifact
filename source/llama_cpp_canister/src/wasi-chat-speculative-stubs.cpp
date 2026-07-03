// Stubs for removed common/ files (chat, speculative, json-schema-to-grammar, etc.)
// These functions are referenced by arg.cpp or other kept files but the full
// implementations are not needed for ICP canister inference.

#include "speculative.h"
#include "chat.h"
#include "json-schema-to-grammar.h"
#include "common.h"

// --- speculative stubs (arg.cpp calls common_speculative_type_to_str) ---

std::string common_speculative_type_name_str() {
    return "none";
}

std::string common_speculative_type_to_str(enum common_speculative_type) {
    return "none";
}

enum common_speculative_type common_speculative_type_from_name(const std::string &) {
    return COMMON_SPECULATIVE_TYPE_NONE;
}

bool common_speculative_is_compat(llama_context *) {
    return false;
}

common_speculative * common_speculative_init(
        common_params_speculative &,
        llama_context *) {
    return nullptr;
}

void common_speculative_free(common_speculative *) {}

void common_speculative_begin(common_speculative *, const llama_tokens &) {}

llama_tokens common_speculative_draft(
                     common_speculative *,
        const common_params_speculative &,
                     const llama_tokens &,
                            llama_token) {
    return {};
}

void common_speculative_accept(common_speculative *, uint16_t) {}

void common_speculative_print_stats(const common_speculative *) {}

// --- chat stubs (arg.cpp calls common_chat_verify_template and common_reasoning_format_from_name) ---

bool common_chat_verify_template(const std::string &, bool) {
    return true;
}

common_reasoning_format common_reasoning_format_from_name(const std::string &) {
    return COMMON_REASONING_FORMAT_NONE;
}

// --- json-schema-to-grammar stubs ---

// Define the incomplete type so unique_ptr destructor works
class common_schema_converter {};

std::string json_schema_to_grammar(const nlohmann::ordered_json &, bool) {
    return "";
}

std::string gbnf_format_literal(const std::string & literal) {
    return literal;
}

std::string build_grammar(const std::function<void(const common_grammar_builder &)> &, const common_grammar_options &) {
    return "";
}

common_schema_info::common_schema_info() : impl_(nullptr) {}
common_schema_info::~common_schema_info() = default;
common_schema_info::common_schema_info(common_schema_info &&) noexcept = default;
common_schema_info & common_schema_info::operator=(common_schema_info &&) noexcept = default;
void common_schema_info::resolve_refs(nlohmann::ordered_json &) {}
bool common_schema_info::resolves_to_string(const nlohmann::ordered_json &) { return false; }
