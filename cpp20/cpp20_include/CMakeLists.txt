add_subdirectory(
    ../core
    ${CMAKE_BINARY_DIR}/cpp20/core
)
add_library(cpp20_include INTERFACE)

target_include_directories(cpp20_include
    INTERFACE
        ${CMAKE_SOURCE_DIR}/cpp20/core
        ${CMAKE_SOURCE_DIR}/cpp20/intrusive_ptr
        ${CMAKE_SOURCE_DIR}/cpp20/utils/core
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(cpp20_include INTERFACE core)

target_sources(cpp20_include
    INTERFACE
        Allocator.hpp
        Device.hpp
        Dtype.hpp
        intrusive_ptr.hpp
        utils.hpp
)
