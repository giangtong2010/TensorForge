#include "_windows_alloc_delete.hpp"

namespace windows_services {
    size_t Windows_Services::get_available_ram() noexcept {
        MEMORYSTATUSEX status;
            status.dwLength = sizeof(status);
            
            if (!GlobalMemoryStatusEx(&status)) {
                return 0;
            }
            return static_cast<size_t>(status.ullAvailPhys);
    }
    void* Windows_Services::windows_allocate(size_t bytes, size_t kMinCPUAlignment) {
        return _aligned_malloc(bytes, kMinCPUAlignment);
    }
    void Windows_Services::windows_free(void* ptr) noexcept {
        _aligned_free(ptr);
    }
}
