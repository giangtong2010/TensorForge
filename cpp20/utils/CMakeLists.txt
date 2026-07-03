add_library(utils INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/core/compute_numel.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/compute_strides.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/is_perm.hpp
)

target_include_directories(utils 
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include 
        ${CMAKE_CURRENT_SOURCE_DIR}/core
)
