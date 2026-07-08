#include "xpu_alloc.hpp"
#include "based_queues.hpp"

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

            auto based_queues = based_queues::instance();
            based_queues.add_queues_and_based(block->_ptr, block->_segment->_queue);

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
    cpp20::Data_ptr XPUAllocator::allocate(size_t nbytes, size_t indx_dev) {
        if (nbytes != 0) {
            size_t alinged_size =
                ((nbytes + kXPUAlignment - 1) / kXPUAlignment) * kXPUAlignment;
            Block* block = _pool.find_free_block(alinged_size, indx_dev);

            auto based_queues = based_queues::instance();
            based_queues.add_queues_and_based(block->_ptr, block->_segment->_queue);

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
