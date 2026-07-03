add_executable(test_cpu_alloc test_cpu_alloc.cpp)
target_link_libraries(test_cpu_alloc
    PRIVATE 
        cpp20_include
        aten_include
)
message(STATUS "Loading cpp20/include")
