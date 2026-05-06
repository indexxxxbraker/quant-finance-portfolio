# CMakeLists.txt
#
# Build configuration for the C++ component of the project.

cmake_minimum_required(VERSION 3.14)
project(quant LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include/quant)

# Library: list every source file here.
add_library(quant_lib
    src/black_scholes.cpp
    src/implied_volatility.cpp
    src/monte_carlo.cpp
    src/gbm.cpp
    src/variance_reduction.cpp
)

add_executable(quant_demo src/main.cpp)
target_link_libraries(quant_demo PRIVATE quant_lib)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(quant_lib  PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(quant_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# Testing with Catch2
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2
)
FetchContent_MakeAvailable(Catch2)

# Test executable: list every test source here.
add_executable(quant_tests
    tests/test_black_scholes.cpp
    tests/test_implied_volatility.cpp
    tests/test_monte_carlo.cpp
    tests/test_variance_reduction.cpp
)
target_link_libraries(quant_tests PRIVATE quant_lib Catch2::Catch2WithMain)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(quant_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_test(NAME black_scholes_tests COMMAND quant_tests)
