list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(SYCL REQUIRED)

add_library(src OBJECT
    choose_allocator.cpp
)

target_include_directories(src
    PRIVATE 
        ${SYCL_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/cpp20/cpp20_include
        ${CMAKE_SOURCE_DIR}/cpp20/core
        ${CMAKE_SOURCE_DIR}/cpp20/device/cpu/cpu_include
        ${CMAKE_SOURCE_DIR}/cpp20/device/cpu/core
        ${CMAKE_SOURCE_DIR}/cpp20/device/xpu/xpu_include
        ${CMAKE_SOURCE_DIR}/cpp20/device/xpu/core
)
