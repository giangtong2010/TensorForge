add_library(xpu_include INTERFACE)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(SYCL REQUIRED)
find_package(LevelZero REQUIRED)

target_include_directories(xpu_include INTERFACE
    ${CMAKE_SOURCE_DIR}/cpp20/device/xpu/core
    ${SYCL_INCLUDE_DIR}
    ${LEVEL_ZERO_INCLUDE_DIR}
)
target_link_libraries(xpu_include INTERFACE
    ${SYCL_LIB_DIR}
    ${LEVEL_ZERO_LIB_DIR}
)

target_sources(xpu_include
    INTERFACE 
        xpu.hpp
)
