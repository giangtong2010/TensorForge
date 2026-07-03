add_library(core STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/choose_allocator.cpp
)

target_link_libraries(core PUBLIC 
    include
    device
)

target_include_directories(core 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
