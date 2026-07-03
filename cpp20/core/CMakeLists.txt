add_library(core STATIC)

target_sources(core
    PRIVATE 
        FILE_SET private_headers TYPE HEADERS
            ${CMAKE_CURRENT_SOURCE_DIR}
        FILES 
            ${CMAKE_CURRENT_SOURCE_DIR}/_allocator.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/_Device.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/_Dtype.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/choose_allocator.cpp
)
