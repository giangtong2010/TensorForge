add_library(intrusive_ptr
INTERFACE 
    intrusive_ptr.hpp
    intrusive_refcounted.hpp
    make_intrusive.hpp
)

target_include_directories(intrusive_ptr PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
