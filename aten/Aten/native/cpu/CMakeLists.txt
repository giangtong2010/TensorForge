add_library(cpu_kernel STATIC 
    src/cpu_binary_ops.cpp
    src/cpu_memory_kn.cpp
    src/cpu_linalg_ops.cpp
    src/cpu_memory_ops.cpp
)

target_link_libraries(cpu_kernel
    PRIVATE 
        cpp20_include
        aten_include
)

target_include_directories(cpu_kernel
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/core
        ${CMAKE_SOURCE_DIR}/aten/aten_include 
        ${CMAKE_SOURCE_DIR}/aten/Aten/core
        ${CMAKE_SOURCE_DIR}/cpp20/core
        ${CMAKE_SOURCE_DIR}/cpp20/intrusive_ptr
        ${CMAKE_SOURCE_DIR}/cpp20/utils/core
        ${CMAKE_SOURCE_DIR}/cpp20/cpp20_include
)
