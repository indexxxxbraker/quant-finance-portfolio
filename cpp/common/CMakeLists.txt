# Library with the common utilities
add_library(quant_common
    normal_distribution.cpp
)

target_include_directories(quant_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Executable that demonstrates the library
add_executable(common_demo main.cpp)
target_link_libraries(common_demo PRIVATE quant_common)