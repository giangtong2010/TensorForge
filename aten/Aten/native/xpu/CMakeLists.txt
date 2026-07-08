list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(FindSYCL)

find_package(SYCL REQUIRED)

add_library(xpu_kernel STATIC 
    src/xpu_kernel.cpp
)

target_link_directories(xpu_kernel
    PRIVATE 
        cpp20_include
        aten_include
        ${SYCL_LIB_DIR}
)

target_include_directories(xpu_kernel
    PRIVATE
        ${CMAKE_SOURCE_DIR}/aten/aten_include 
        ${CMAKE_SOURCE_DIR}/aten/Aten/core
        ${CMAKE_SOURCE_DIR}/cpp20/core
        ${SYCL_INCLUDE_DIR}

    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/core
)
