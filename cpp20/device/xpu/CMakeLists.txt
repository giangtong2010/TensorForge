list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(FindSYCL)
include(FindLevelZero)

find_package(SYCL REQUIRED)
find_package(LevelZero REQUIRED)

add_library(xpu STATIC
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
        ${LEVEL_ZERO_INCLUDE_DIR}/level_zero
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/xpu_include
)
target_link_libraries(xpu 
    PRIVATE 
        cpp20_include
        ${SYCL_LIB_DIR}
        ${LEVEL_ZERO_LIB_DIR}
    PUBLIC xpu_include
)
