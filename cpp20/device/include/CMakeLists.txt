add_library(device_include INTERFACE)

target_include_directories(device_include INTERFACE 
    ${CMAKE_SOURCE_DIR}/cpp20/device/cpu/core
    ${CMAKE_SOURCE_DIR}/cpp20/device/xpu/core
)

target_sources(device_include INTERFACE 
    _cpu.hpp
    _xpu.hpp
)
