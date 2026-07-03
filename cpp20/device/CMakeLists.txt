add_library(device STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/xpu/core/src/get_free_vram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/xpu/core/src/xpu_alloc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/xpu/core/src/xpu_caching.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/core/src/_windows_services.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/core/src/cpu_alloc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/core/src/cpu_caching.cpp
)

target_link_libraries(device PUBLIC 
    include
)

target_include_directories(device 
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/cpp20/device/cpu/core
        ${CMAKE_SOURCE_DIR}/cpp20/device/xpu/core
        ${CMAKE_SOURCE_DIR}/cpp20/incldue
        "D:/Intel/level-zero-1.32.0/include"
    PUBLIC 
        ${CMAKE_SOURCE_DIR}/cpp20/device/cpu/include
        ${CMAKE_SOURCE_DIR}/cpp20/device/xpu/include
)
