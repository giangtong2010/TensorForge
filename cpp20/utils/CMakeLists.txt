add_library(utils STATIC)

target_sources(utils
    PRIVATE 
        FILE_SET private_headers TYPE HEADERS
            ${CMAKE_CURRENT_SOURCE_DIR}/core
        FILES 
            ${CMAKE_CURRENT_SOURCE_DIR}/core/compute_numel.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/core/compute_strides.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/core/is_perm.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/core/src/compute_numel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/core/src/compute_strides.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/core/src/is_perm.cpp
)

target_include_directories(utils 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
