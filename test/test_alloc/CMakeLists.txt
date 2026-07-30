list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(FindSYCL)

add_executable(
    test_xpu_alloc test_xpu_alloc.cpp
)
add_executable(
    test_cpu_alloc test_cpu_alloc.cpp
)
add_executable(
    test_sycl test_sycl.cpp
)

target_link_libraries(test_xpu_alloc
    PRIVATE 
        cpp20
        aten
        ${SYCL_LIB_DIR}
)
target_link_libraries(test_cpu_alloc 
    PRIVATE 
        cpp20
        aten
        ${SYCL_LIB_DIR}
)
target_link_libraries(test_sycl PRIVATE 
    ${SYCL_LIB_DIR}
)
