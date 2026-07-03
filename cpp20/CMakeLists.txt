add_library(cpp20 INTERFACE)
add_subdirectory(device)
add_subdirectory(utils)
add_subdirectory(cpp20_include)

target_sources(cpp20 PRIVATE 
    $<TARGET_OBJECTS:utils>
)

target_link_libraries(cpp20 
    INTERFACE 
        device
        cpp20_include
)

target_include_directories(cpp20 INTERFACE
    ${CMAKE_SOURCE_DIR}/cpp20/include
    ${CMAKE_SOURCE_DIR}/cpp20/core
)
