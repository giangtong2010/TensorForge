add_library(core INTERFACE)
target_sources(core INTERFACE 
    _allocator.hpp
    _Device.hpp
    _Dtype.hpp
)

target_include_directories(core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
