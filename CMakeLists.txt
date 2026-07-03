cmake_minimum_required(VERSION 4.2)
project(TensorForge
    VERSION 1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(aten)
add_subdirectory(cpp20)
