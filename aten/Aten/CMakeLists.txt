add_library(Aten STATIC)

target_sources(Aten
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/core/src/_memory_layout.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/core/src/_reduction_ops.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/core/src/_shape_ops.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/core/src/_view_ops.cpp

        FILE_SET private_headers TYPE HEADERS
        BASE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/core
        FILES 
            ${CMAKE_CURRENT_SOURCE_DIR}/core/_storage_impl.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/core/_storage.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/core/_tensor_impl.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/core/_tensor.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/core/_tensorBody.hpp

    PUBLIC 
        FILE_SET public_headers TYPE HEADERS
        BASE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        FILES

            ${CMAKE_CURRENT_SOURCE_DIR}/include/tensor.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/impl/storage_impl.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/impl/storage.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/impl/tensor_impl.hpp
)

target_include_directories(Aten 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/core
)
