#pragma once
#include <cstdint>
#include <new>
#include <malloc.h>
#include <vector>
#include <windows.h>

#ifdef _WIN32
namespace windows_services {
    class Windows_Services {
    public:
        static void* windows_allocate(size_t nbytes, size_t kMinCPUAlignment);
        static void windows_free(void* ptr) noexcept;
        static size_t get_available_ram() noexcept;
    };
}
#endif
