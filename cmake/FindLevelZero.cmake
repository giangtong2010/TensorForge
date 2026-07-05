find_path(LEVEL_ZERO_INCLUDE_DIR
    NAMES 
        level_zero/zes_api.h
        level_zero/ze_api.h
    HINTS ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES include
)
find_library(LEVEL_ZERO_LIB_DIR
    NAMES ze_loader.lib
    HINTS ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib
)

message(STATUS "LEVEL_ZERO_INCLUDE_DIR = ${LEVEL_ZERO_INCLUDE_DIR}")
message(STATUS "LEVEL_ZERO_LIB_DIR = ${LEVEL_ZERO_LIB_DIR}")
