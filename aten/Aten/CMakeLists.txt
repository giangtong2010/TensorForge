add_library(Aten OBJECT
    src/_memory_layout.cpp
    src/_reduction_ops.cpp
    src/_shape_ops.cpp
    src/_view_ops.cpp
)

target_include_directories(Aten PRIVATE 
    ${CMAKE_SOURCE_DIR}/aten/Aten/core
    ${CMAKE_SOURCE_DIR}/aten/aten_include
    ${CMAKE_SOURCE_DIR}/cpp20/cpp20_include
)

target_link_libraries(Aten PRIVATE 
    cpp20_include
    aten_include
)
