find_path(SYCL_INCLUDE_DIR
    NAMES sycl/sycl.hpp
    PATH_SUFFIXES include
)

find_path(LEVEL_ZERO_DIR
    NAMES
        ze_api.h
        zes_api.h
    HINTS "D:/Intel/level-zero-1.32.0"
    PATH_SUFFIXES include
)

add_library(xpu OBJECT
    src/get_free_vram.cpp
    src/xpu_alloc.cpp
    src/xpu_caching.cpp
)

target_include_directories(xpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_SOURCE_DIR}/cpp20/cpp20_include
    ${SYCL_INCLUDE_DIR}
    ${LEVEL_ZERO_DIR}
)
target_link_libraries(xpu PRIVATE cpp20_include)