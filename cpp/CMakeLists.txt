# CMakeLists.txt
#
# Build configuration for the C++ component of the project.

cmake_minimum_required(VERSION 3.14)
project(quant LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include/quant)

add_library(quant_lib src/black_scholes.cpp)
add_executable(quant_demo src/main.cpp)
target_link_libraries(quant_demo PRIVATE quant_lib)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(quant_lib  PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(quant_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# Testing with Catch2
# ---------------------------------------------------------------------------
# FetchContent downloads dependencies at configure time, so we don't have to
# install Catch2 system-wide. The first `cmake ..` will clone the repo
# (one-time cost ~30s), subsequent builds reuse the cached copy.

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2  # Pin to a known-good version for reproducibility.
)
FetchContent_MakeAvailable(Catch2)

# The test executable. Catch2 v3 provides its own main() via the
# Catch2WithMain target, so we don't write one ourselves.
add_executable(quant_tests tests/test_black_scholes.cpp)

# Link the test executable against our library and Catch2's main.
target_link_libraries(quant_tests PRIVATE quant_lib Catch2::Catch2WithMain)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(quant_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Register the test executable with CTest, CMake's built-in test runner.
enable_testing()
add_test(NAME black_scholes_tests COMMAND quant_tests)
