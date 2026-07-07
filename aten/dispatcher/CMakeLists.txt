add_library(dispatcher OBJECT
    register.cpp
)

target_include_directories(dispatcher PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/aten/Aten/core
        ${CMAKE_SOURCE_DIR}/aten
)

target_link_libraries(dispatcher PRIVATE 
    cpp20_include
    aten_include
)
