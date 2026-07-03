add_library(include INTERFACE
    Allocator.hpp
    backend.hpp
    Device.hpp
    Dtype.hpp
)

target_include_directories(include INTERFACE
    ${CMAKE_SOURCE_DIR}/cpp20/include
)
