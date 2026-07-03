add_library(Aten STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/_memory_layout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/_reduction_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/_shape_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/_view_ops.cpp
) 

target_include_directories(Aten PUBLIC
    include
    cpp20/include
    intrusive_ptr
)

target_include_directories(Aten 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/core
)
