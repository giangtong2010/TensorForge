add_library(cpp20 SHARED)
add_subdirectory(utils)
add_subdirectory(src)
add_subdirectory(device/cpu)
add_subdirectory(device/xpu)
add_subdirectory(cpp20_include)

target_sources(cpp20 PRIVATE 
    $<TARGET_OBJECTS:utils>
    $<TARGET_OBJECTS:src>
    $<TARGET_OBJECTS:cpu>
)

target_link_libraries(cpp20 
    PUBLIC
        cpp20_include
)

target_include_directories(cpp20 
    PUBLIC
        ${CMAKE_SOURCE_DIR}/cpp20/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp20/core
        cpp20_include
)
