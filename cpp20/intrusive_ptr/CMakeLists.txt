add_library(intrusive_ptr
INTERFACE
    intrusive_ptr.hpp
    intrusive_refcounted.hpp
    make_intrusive.hpp
)

target_include_directories(intrusive_ptr INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp20/intrusive_ptr/include
)
