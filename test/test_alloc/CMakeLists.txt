add_executable(test_cpu_alloc test_cpu_alloc.cpp)
add_subdirectory(
    ${CMAKE_SOURCE_DIR}/cpp20/device/cpu
    ${CMAKE_BINARY_DIR}/cpp20/device/cpu
)

target_link_libraries(test_cpu_alloc
    PRIVATE 
        cpp20
        aten
        cpu
)
message(STATUS "Loading cpp20/include")
