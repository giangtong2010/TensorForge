list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(FindSYCL)

add_executable(
    test_tensor test_tensor.cpp
)

target_link_libraries(test_tensor
    PRIVATE 
        cpp20
        aten
        ${SYCL_LIB_DIR}
)
