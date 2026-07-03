add_library(cpu OBJECT
    src/_windows_services.cpp
    src/cpu_alloc.cpp
    src/cpu_caching.cpp
)

target_include_directories(cpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_SOURCE_DIR}/cpp20/cpp20_include
)

target_link_libraries(cpu PRIVATE cpp20_include)
