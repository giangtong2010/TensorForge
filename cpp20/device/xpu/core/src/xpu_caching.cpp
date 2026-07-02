#include "xpu_caching.hpp"
#include <sycl/sycl.hpp>
#include <utility>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
size_t get_available_ram() {
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) {
        return 0;
    }
    return static_cast<size_t>(status.ullAvailPhys);
}
#endif

// ======================== BLOCK ============================
namespace xpu {
    void Block::mark_allocated() {
        if (!allocated) {
            allocated = true;
            _segment->allocated_bytes += _bytes;
            _segment->active_block += 1;
        }
    }
    void Block::mark_free() {
        if (allocated) {
            allocated = false;
            _segment->allocated_bytes -= _bytes;
            _segment->active_block -= 1;
        }
    }

// =============================== POOL ==================================
    Pool::Pool() {
        auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
        for (size_t i = 0; i < gpus.size(); i++) {
            sycl::queue queue(gpus[i]);
            _queues.push_back(queue);
        }
    }

    Segment* Pool::allocate_segment(size_t nbytes) {
        auto allocate = [=](size_t bytes) {
            uint8_t* _ptr = sycl::malloc_device<uint8_t>(bytes, _queues[indx_alloc]);
            if (!_ptr) {
                free_mem();
                _ptr = sycl::malloc_device<uint8_t>(bytes, _queues[indx_alloc]);
            }

            Segment* segment = new Segment {
                _queues[indx_alloc],
                _ptr,
                bytes,
            };
            Block* head = new Block {
                _ptr,
                bytes,
                false,
                nullptr,
                nullptr,
                segment
            };
            segment->_head = head;

            _free_block.push_back(head);
            _segment.push_back(segment);
            for (size_t i = 0; i < _segment.size(); i++) {
                if (!_segment[i]->is_free()) {
                    active_bytes += _segment[i]->allocated_bytes;
                    active_block += _segment[i]->active_block;
                }
            }
            if (active_bytes - (128 * 1024 * 1024) > available_ram) {
                available_ram = get_available_ram();
            }
            return segment;
        };
        Segment* segment = nullptr;
        size_t aligned_size = 
            ((nbytes + kMinXPUAlignment - 1) / kMinXPUAlignment) * kMinXPUAlignment;

        if (indx_alloc < _queues.size()) {
            segment = allocate(aligned_size);
            indx_alloc += 1;
        }
        else {
            indx_alloc = 0;
            segment = allocate(aligned_size);
            indx_alloc += 1;
        };
        
        return segment;
    }

    Block* Pool::find_free_block(size_t request_size) {
        if (!_free_block.empty()) {
            for (size_t i = 0; i < _free_block.size(); i++) {
                auto block = _free_block[i];
                if (block->_bytes >= request_size) {
                    _free_block.erase(_free_block.begin() + i);
                    auto allocated_block = split(request_size, block);
                    return allocated_block;
                }
            }
        }
        auto segment = allocate_segment(request_size);
        return split(request_size, segment->_head);
    }
    Block* Pool::split(size_t request_size, Block* block) {
        if (block->_bytes - request_size > kMinBlockSize) {
            size_t remain_bytes = block->_bytes - request_size;
            uint8_t* new_ptr = block->_ptr + request_size;

            Block* new_block = new Block {
                new_ptr,
                remain_bytes,
                false,
                block,
                block->next,
                block->_segment
            };
            if (block->next) {
                block->next->prev = new_block;
            }
            block->next = new_block;

            _free_block.push_back(new_block);
        }
        block->mark_allocated();
        return block;
    }
    Block* Pool::merge(Block* a, Block* b) {
        if (
            !(a->is_allocated() && b->is_allocated()) &&
            (a->_ptr + a->_bytes == b->_ptr)
        ) {
            size_t new_size = a->_bytes + b->_bytes;
            a->next = b->next;

            if (b->next)
                b->next->prev = a->next;
            _free_block.erase(
                std::remove(_free_block.begin(), _free_block.end(), b),
                _free_block.end()
            );
            delete b;
            return a;
        }
        else {
            throw std::invalid_argument(
                "Two argument for merge do not continuous in memory"
            );
        }
    }
    
    void Pool::delete_block(Block* block) {
        if (!block) return;

        block->mark_free();
        if (block->next && !block->next->is_allocated()) {
            block = merge(block, block->next);
        }
        if (block->prev && !block->prev->is_allocated()) {
            block = merge(block->prev, block);
        }

        _free_block.push_back(block);
        available_ram = get_available_ram();
        if (active_bytes > available_ram) {
            free_mem();
        }
    }
    void Pool::free_mem() {
        for (size_t i = 0; i < _segment.size(); i++) {
            auto segment = _segment[i];
            if (segment->is_free()) {
                auto queue = segment->_queue;
                auto head = segment->_head;

                while (head) {
                    _free_block.erase(
                        std::remove(_free_block.begin(), _free_block.end(), head),
                        _free_block.end()
                    );

                    Block* next = head->next;
                    delete head;
                    head = next;
                }
                _segment.erase(
                    std::remove(_segment.begin(), _segment.end(), segment),
                    _segment.end()
                );
                sycl::free(segment->_base_ptr, queue);
                delete segment;
            }
        }
    }
}
