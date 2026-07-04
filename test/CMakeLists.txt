add_library(test INTERFACE)
add_subdirectory(test_alloc)

target_link_libraries(test INTERFACE test_alloc)
