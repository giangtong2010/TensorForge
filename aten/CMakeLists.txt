add_subdirectory(Aten)
add_subdirectory(dispatcher)
add_subdirectory(aten_include)
add_library(aten STATIC)

target_sources(aten PRIVATE 
    $<TARGET_OBJECTS:Aten>
    $<TARGET_OBJECTS:dispatcher>
)
target_link_libraries(aten INTERFACE
    aten_include
)

target_include_directories(aten INTERFACE
    ${CMAKE_SOURCE_DIR}/aten/include
    ${CMAKE_SOURCE_DIR}/aten/dispatcher
)
