add_library(utils OBJECT
    src/compute_numel.cpp
    src/compute_strides.cpp
    src/is_perm.cpp
)

target_include_directories(utils PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/core
)
