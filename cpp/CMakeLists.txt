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
    src/qmc.cpp
    src/greeks.cpp
    src/asian.cpp
    src/american.cpp
    src/pde.cpp
    src/ftcs.cpp
    src/thomas.cpp
    src/btcs.cpp
    src/theta_scheme.cpp
    src/cn.cpp
    src/psor.cpp
    src/cn_american.cpp
    src/trinomial.cpp
    src/heston_fourier.cpp
    src/heston_mc.cpp
    src/heston_qe.cpp
    src/heston_pde.cpp
)

# QMC needs to know the path to the Joe-Kuo direction-numbers file.
# We bake the absolute path in at compile time as a macro. If the
# data file is moved, only this line needs to change.
set(SOBOL_DATA_FILE_PATH "${CMAKE_SOURCE_DIR}/data/new-joe-kuo-6.21201")
target_compile_definitions(quant_lib PUBLIC
    SOBOL_DATA_FILE="${SOBOL_DATA_FILE_PATH}"
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
    tests/test_qmc.cpp
    tests/test_greeks.cpp
    tests/test_asian.cpp
    tests/test_american.cpp
    tests/test_pde.cpp
    tests/test_ftcs.cpp
    tests/test_thomas.cpp
    tests/test_btcs.cpp
    tests/test_theta_scheme.cpp
    tests/test_cn.cpp
    tests/test_psor.cpp
    tests/test_cn_american.cpp
    tests/test_trinomial.cpp
    tests/test_heston_fourier.cpp
    tests/test_heston_mc.cpp
    tests/test_heston_qe.cpp
    tests/test_heston_pde.cpp
)
target_link_libraries(quant_tests PRIVATE quant_lib Catch2::Catch2WithMain)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(quant_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_test(NAME black_scholes_tests COMMAND quant_tests)
