#include "xpu_alloc.hpp"

namespace xpu {
    void XPUAllocator::free_tensor(uint8_t* data, void* ctx) noexcept {
        auto block = static_cast<Block*>(ctx);
        XPUAllocator::instance()._pool.delete_block(block);
    }

    cpp20::Data_ptr XPUAllocator::allocate(size_t nbytes) {
        if (nbytes != 0) {
            size_t alinged_size =
                ((nbytes + kXPUAlignment - 1) / kXPUAlignment) * kXPUAlignment;
            Block* block = _pool.find_free_block(alinged_size);
            return cpp20::Data_ptr {
                block->_ptr,
                static_cast<void*>(block),
                &XPUAllocator::free_tensor
            };
        }
        else {
            return cpp20::Data_ptr {nullptr, nullptr, nullptr};
        }
    }
}
