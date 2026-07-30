add_library(test INTERFACE)
add_subdirectory(test_alloc)
add_subdirectory(test_tensor)

target_link_libraries(test 
    INTERFACE 
        test_alloc
        test_tensor
)
