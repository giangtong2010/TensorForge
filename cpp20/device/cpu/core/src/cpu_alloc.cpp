#include "cpu_alloc.hpp"

namespace cpu {
    void CPUAllocator::free_tensor(uint8_t* data, void* ctx) noexcept {
        auto block = static_cast<Block*>(ctx);
        CPUAllocator::instance()._pool.delete_block(block);
    }

    cpp20::Data_ptr CPUAllocator::allocate(size_t nbytes) {
        if (nbytes != 0) {
            size_t aligned_size = 
                ((nbytes + kCPUAligment - 1) / kCPUAligment) * kCPUAligment;
            Block* block = _pool.find_free_block(aligned_size);
            return cpp20::Data_ptr {
                block->_ptr,
                static_cast<void*>(block),
                &CPUAllocator::free_tensor
            };
        }
        else {
            return cpp20::Data_ptr(
                nullptr, nullptr, nullptr
            );
        }
    }
}
