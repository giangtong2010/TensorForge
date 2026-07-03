add_library(device STATIC)

target_sources(device 
    PRIVATE 
        FILE_SET private_headers TYPE HEADERS
            ${CMAKE_CURRENT_SOURCE_DIR}/cpu/core
            ${CMAKE_CURRENT_SOURCE_DIR}/xpu/core
        FILES 
            ${CMAKE_CURRENT_SOURCE_DIR}/cpu/core/_windows_alloc_delete.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cpu/core/cpu_alloc.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cpu/core/cpu_caching.hpp

            ${CMAKE_CURRENT_SOURCE_DIR}/xpu/core/get_free_ram.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/xpu/core/xpu_alloc.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/xpu/core/xpu_caching.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/xpu/core/src/get_free_ram.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/xpu/core/src/xpu_alloc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/xpu/core/src/xpu_caching.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/cpu/core/src/_windows_services.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu/core/src/cpu_alloc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu/core/src/cpu_caching.cpp
)

target_include_directories(device 
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu/core
        ${CMAKE_CURRENT_SOURCE_DIR}/xpu/core
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu/include
        ${CMAKE_CURRENT_SOURCE_DIR}/xpu/include
)
