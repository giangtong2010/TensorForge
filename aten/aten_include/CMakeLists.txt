add_library(aten_include INTERFACE)

target_include_directories(aten_include INTERFACE 
    ${CMAKE_SOURCE_DIR}/aten/dispatcher
    ${CMAKE_SOURCE_DIR}/aten/Aten/core
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(aten_include INTERFACE 
    dispatcher/at_dispatcher_kernel.hpp
    dispatcher/dispatcher.hpp
    dispatcher/register_kernel.hpp

    impl/storage_impl.hpp
    impl/storage.hpp
    impl/tensor_impl.hpp

    tensor.hpp
)
