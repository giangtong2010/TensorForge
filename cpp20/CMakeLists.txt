add_library(cpp20 INTERFACE)

target_link_libraries(cpp20 
INTERFACE 
    core
    inc
    intrusive_ptr
    utils
    device
)
