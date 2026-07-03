add_subdirectory(dispatcher)
add_subdirectory(Aten)
add_library(aten INTERFACE)

target_link_libraries(aten 
INTERFACE
    dispatcher
    Aten
)
