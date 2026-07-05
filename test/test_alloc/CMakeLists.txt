add_executable(test_xpu_alloc test_xpu_alloc.cpp)

target_link_libraries(test_xpu_alloc
    PRIVATE 
        cpp20
        aten
)
message(STATUS "Loading cpp20/include")
