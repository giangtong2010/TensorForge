add_library(inc INTERFACE
    Allocator.hpp
    backend.hpp
    Device.hpp
    Dtype.hpp
)

target_include_directories(inc PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)