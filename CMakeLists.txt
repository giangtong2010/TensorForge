cmake_minimum_required(VERSION 4.2)
project(TensorForge
    VERSION 4.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(aten)
add_subdirectory(cpp20)
add_subdirectory(test)
