#pragma once
#include <sycl/sycl.hpp>
#include <cstddef>
#include "xpu_caching.hpp"
#include "Allocator.hpp"

namespace xpu {
    class XPUAllocator : public cpp20::Allocator {
        const size_t kXPUAlignment = 64;
        Pool _pool;
        static void free_tensor(uint8_t* data, void* ctx) noexcept;

    public:
        XPUAllocator() = default;
        XPUAllocator(const XPUAllocator&) = delete;
        XPUAllocator& operator=(const XPUAllocator&) = delete;

        cpp20::Data_ptr allocate(size_t nbytes) override;
        
        static XPUAllocator& instance() {
            static XPUAllocator allocator;
            return allocator;
        }
    };
}
