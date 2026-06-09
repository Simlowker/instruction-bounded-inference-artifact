cmake_minimum_required(VERSION 3.14)
project("native-bench" C CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Path to fork
set(FORK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../llama_cpp_canister/src/llama_cpp_onicai_fork")
get_filename_component(FORK_DIR "${FORK_DIR}" ABSOLUTE)

# The fork's sub-CMakeLists use llama_add_compile_flags() — define a no-op
macro(llama_add_compile_flags)
endmacro()

# Versions expected by src/CMakeLists.txt
set(LLAMA_INSTALL_VERSION "0.0.0")

# ggml (standalone sub-project)
add_subdirectory("${FORK_DIR}/ggml" "${CMAKE_BINARY_DIR}/ggml")

# llama library
add_subdirectory("${FORK_DIR}/src" "${CMAKE_BINARY_DIR}/llama")

# Benchmark binary — uses only the llama C API (no common/ dependency)
add_executable(bench bench.cpp)
target_include_directories(bench PRIVATE "${FORK_DIR}/include")
target_link_libraries(bench PRIVATE llama ggml)
