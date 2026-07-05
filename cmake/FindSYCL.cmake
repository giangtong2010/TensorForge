find_path(SYCL_INCLUDE_DIR
    NAMES sycl/sycl.hpp
    HINTS
        ENV CMPLR_ROOT
    PATH_SUFFIXES include
)
find_library(SYCL_LIB_DIR
    NAMES sycl.lib
    HINTS
        ENV ONEAPI_ROOT
    PATH_SUFFIXES compiler/latest/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SYCL
    REQUIRED_VARS
        SYCL_INCLUDE_DIR
        SYCL_LIB_DIR
)

message(STATUS "SYCL_INCLUDE_DIR = ${SYCL_INCLUDE_DIR}")
message(STATUS "SYCL_LIB_DIR = ${SYCL_LIB_DIR}")
