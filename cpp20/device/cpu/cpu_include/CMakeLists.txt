add_library(cpu_include INTERFACE)

target_include_directories(cpu_include INTERFACE 
    ${CMAKE_SOURCE_DIR}/cpp20/device/cpu/core
)

target_sources(cpu_include
    INTERFACE 
        cpu.hpp
)
