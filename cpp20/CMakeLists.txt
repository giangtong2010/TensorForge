add_subdirectory(core)
add_subdirectory(include)
add_subdirectory(intrusive_ptr)
add_subdirectory(utils)
add_subdirectory(device)
add_library(cpp20 INTERFACE)

target_link_libraries(cpp20 
INTERFACE
    core
    inc
    intrusive_ptr
    utils
    device
)
