add_library(xpu_include INTERFACE)

find_path(SYCL_INCLUDE_DIR
    NAMES sycl/sycl.hpp
    HINTS
        ENV SYCL_ROOT
        ENV ONEAPI_ROOT
        ${CMAKE_PREFIX_PATH}
)
find_library(SYCL_LIB_DIR
    NAMES sycl
    HINTS
        ENV SYCL_ROOT
        ENV ONEAPI_ROOT
)

find_path(LEVEL_ZERO_DIR
    NAMES ze_api.h
)
find_library(LEVEL_ZERO_LIB
    NAMES ze_loader    
)

target_include_directories(xpu_include INTERFACE
    ${CMAKE_SOURCE_DIR}/cpp20/device/xpu/core
    ${SYCL_INCLUDE_DIR}
    ${LEVEL_ZERO_DIR}
)
target_link_libraries(xpu_include INTERFACE
    ${SYCL_LIB_DIR}
    ${LEVEL_ZERO_LIB}
)

target_sources(xpu_include
    INTERFACE 
        xpu.hpp
)
