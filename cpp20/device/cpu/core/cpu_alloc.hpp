#pragma once
#include <new>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <malloc.h>
#include "cpu_caching.hpp"
#include "Allocator.hpp"

namespace cpu {
    class CPUAllocator : public cpp20::Allocator {
        const size_t kCPUAligment = 64;
        Pool _pool;
        static void free_tensor(uint8_t* data, void* ctx) noexcept;

    public:
        CPUAllocator() = default;
        CPUAllocator(const CPUAllocator&) = delete;
        CPUAllocator& operator=(const CPUAllocator&) = delete;
        
        cpp20::Data_ptr allocate(size_t nbytes) override;

        static CPUAllocator& instance() {
            static CPUAllocator allocator;
            return allocator;
        }
    };
};
