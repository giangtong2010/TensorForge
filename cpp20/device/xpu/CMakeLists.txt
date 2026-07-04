add_library(xpu OBJECT
    src/get_free_vram.cpp
    src/xpu_alloc.cpp
    src/xpu_caching.cpp
)
add_subdirectory(xpu_include)

target_include_directories(xpu 
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core
        ${CMAKE_SOURCE_DIR}/cpp20/cpp20_include
        ${SYCL_INCLUDE_DIR}
        ${LEVEL_ZERO_DIR}
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/xpu_include
)
target_link_libraries(xpu 
    PRIVATE cpp20_include
    PUBLIC xpu_include
)
