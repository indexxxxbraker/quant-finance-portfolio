cmake_minimum_required(VERSION 3.20)

project(QuantFinancePortfolio
    VERSION 0.1.0
    DESCRIPTION "Pricing of financial derivatives in C++"
    LANGUAGES CXX
)

# C++ standard: C++20 (modern, well-supported, gives us nice features)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Enable strict warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Add subdirectories with code
add_subdirectory(common)