add_library(device INTERFACE)
add_subdirectory(cpu)
add_subdirectory(xpu)
add_subdirectory(include)

target_sources(device INTERFACE 
    $<TARGET_OBJECTS:cpu> 
    $<TARGET_OBJECTS:xpu>
)

target_link_libraries(device INTERFACE 
    device_include
)

target_include_directories(device INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
