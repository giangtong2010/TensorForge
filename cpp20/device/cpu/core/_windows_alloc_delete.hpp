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
        static void* windows_allocate(size_t nbytes, size_t kMinCPUAlignment) {
            void* ptr = _aligned_malloc(nbytes, kMinCPUAlignment);
            return ptr;
        }
        static void windows_free(void* ptr) noexcept {
            _aligned_free(ptr);
        }
        static size_t get_available_ram() noexcept {
            MEMORYSTATUSEX status;
            status.dwLength = sizeof(status);
            
            if (!GlobalMemoryStatusEx(&status)) {
                return 0;
            }
            return static_cast<size_t>(status.ullAvailPhys);
        }
    };
}
#endif
