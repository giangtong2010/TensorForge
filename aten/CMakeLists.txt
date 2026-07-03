add_subdirectory(Aten)
add_subdirectory(dispatcher)
add_subdirectory(aten_include)
add_library(aten INTERFACE)

target_sources(aten PRIVATE 
    $<TARGET_OBJECTS:Aten>
)
target_link_libraries(aten INTERFACE
    dispatcher
    aten_include
)

target_include_directories(aten INTERFACE
    ${CMAKE_SOURCE_DIR}/aten/include
)
