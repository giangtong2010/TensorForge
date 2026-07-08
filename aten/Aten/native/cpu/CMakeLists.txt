add_library(cpu_kernel STATIC 
    src/cpu_kernel.cpp
)

target_link_libraries(cpu_kernel
    PRIVATE 
        cpp20_include
        ate_include
)

target_include_directories(cpu_kernel
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/core
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/aten/aten_include 
        ${CMAKE_SOURCE_DIR}/aten/Aten/core
        ${CMAKE_SOURCE_DIR}/cpp20/core
        ${CMAKE_SOURCE_DIR}/cpp20/cpp20_include
)
