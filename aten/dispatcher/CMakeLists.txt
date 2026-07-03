add_library(dispatcher INTERFACE
    _at_dispatcher_kernel.hpp
    _dispatcher.hpp
    _register_kernel.hpp
    _table.hpp
)

target_include_directories(dispatcher INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}
)
