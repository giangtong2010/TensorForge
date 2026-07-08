#include "xpu_caching.hpp"
#include "get_free_vram.hpp"
#include "based_queues.hpp"
#include <sycl/sycl.hpp>
#include <utility>
#include <stdexcept>
#include <algorithm>

size_t get_available_ram(const sycl::device& device) {
    return get_free_vram(device);
}

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
        if (gpus.empty()) {
            throw std::runtime_error(
                "cannot find Intel GPU on your device, try using different backend like cpu"
            );
        }
        for (size_t i = 0; i < gpus.size(); i++) {
            sycl::queue queue(gpus[i]);
            _queues.push_back(queue);
            _device.emplace_back(std::move(gpus[i]));

            auto based_queues = based_queues::instance();
            based_queues.add_queue_and_indx(queue, i);
        }
    }

    Segment* Pool::allocate_segment(size_t nbytes, size_t indx_dev) {
        if (indx_dev >= _queues.size()) {
            throw std::invalid_argument(
                "indx of devices is out of the number devices have"
            );
        }

        size_t aligned_size = 
            ((nbytes + kMinXPUAlignment - 1) / kMinXPUAlignment) * kMinXPUAlignment;

        int chance = 5;
        uint8_t* _ptr = sycl::malloc_device<uint8_t>(aligned_size, _queues[indx_dev]);
        if (!_ptr) {
            while (chance-- > 0) {
                free_mem(indx_dev);
                _ptr = sycl::malloc_device<uint8_t>(aligned_size, _queues[indx_dev]);
                if (_ptr) break;
            }
            if (!_ptr)
                throw std::bad_alloc();
            }

        try {
            Segment* segment = new Segment {
                _queues[indx_dev],
                _ptr,
                aligned_size,
            };
            Block* head = new Block {
                _ptr,
                aligned_size,
                false,
                nullptr,
                nullptr,
                segment
            };

            segment->_head = head;
            _free_block.insert({aligned_size, head});
            _segment.push_back(segment);

            auto& gpu = _device[indx_dev];
            gpu.allocated_bytes += segment->_total_bytes;
            if (gpu.allocated_bytes - (128 * 1024 * 1024) > gpu.free_vram) {
                gpu.free_vram = get_available_ram(gpu._device);
            }
            return segment;

        } catch(...) {
            sycl::free(_ptr, _queues[indx_dev]);
            throw;
        }
    }

    Segment* Pool::allocate_segment(size_t nbytes) {
        auto allocate = [this](size_t bytes) {
            int chance = 5;
            uint8_t* _ptr = sycl::malloc_device<uint8_t>(bytes, _queues[indx_alloc]);
            if (!_ptr) {
                while (chance-- > 0) {
                    free_mem(indx_alloc);
                    _ptr = sycl::malloc_device<uint8_t>(bytes, _queues[indx_alloc]);
                    if (_ptr) break;
                }
                if (!_ptr)
                    throw std::bad_alloc();
            }

            try {
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
                _free_block.insert({bytes, head});
                _segment.push_back(segment);

                auto& gpu = _device[indx_alloc];
                gpu.allocated_bytes += segment->_total_bytes;
                if (gpu.allocated_bytes - (128 * 1024 * 1024) > gpu.free_vram) {
                    gpu.free_vram = get_available_ram(gpu._device);
                }
                return segment;
            } catch(...) {
                sycl::free(_ptr, _queues[indx_alloc]);
                throw;
            }
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
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _free_block.lower_bound(request_size);

        if (it != _free_block.end()) {
            auto block = it->second;
            if (!block->allocated && block->_bytes >= request_size) {
                _free_block.erase(it);
                return split(request_size, block);
            }
        }
        auto segment = allocate_segment(request_size);
        return split(request_size, segment->_head);
    }
    Block* Pool::find_free_block(size_t request_size, size_t indx_dev) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (indx_dev >= _queues.size()) {
            throw std::invalid_argument(
                "indx of devices is out of the number devices have"
            );
        }
        
        auto it = _free_block.lower_bound(request_size);

        if (it != _free_block.end()) {
            auto block = it->second;
            if (
                !block->allocated && 
                block->_bytes >= request_size &&
                block->_segment->_queue == _queues[indx_dev]
            ) {
                _free_block.erase(it);
                return split(request_size, block);
            }
        }
        auto segment = allocate_segment(request_size, indx_dev);
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
            block->_bytes = request_size;

            if (block->next) {
                block->next->prev = new_block;
            }
            block->next = new_block;

            _free_block.insert({remain_bytes, new_block});
        }
        block->mark_allocated();
        return block;
    }
    Block* Pool::merge(Block* a, Block* b) {
        if (
            !a->is_allocated() && !b->is_allocated() &&
            a->_ptr + a->_bytes == b->_ptr &&
            a->_segment == b->_segment
        ) {
            size_t new_size = a->_bytes + b->_bytes;
            a->_bytes = new_size;
            a->next = b->next;

            if (b->next)
                b->next->prev = a;

            for (auto it = _free_block.lower_bound(b->_bytes);
                it != _free_block.upper_bound(b->_bytes);
                it++) {
                    if (it->second == b) {
                        _free_block.erase(it);
                        break;
                    }
                }
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
        std::lock_guard<std::mutex> lock(_mutex);
        if (!block) return;

        block->mark_free();
        if (block->next && !block->next->is_allocated()) {
            block = merge(block, block->next);
        }
        if (block->prev && !block->prev->is_allocated()) {
            block = merge(block->prev, block);
        }

        _free_block.insert({block->_bytes, block});
        auto device = block->_segment->_queue.get_device();
        auto gpu_it = std::find_if(
            _device.begin(),
            _device.end(),
            [&](const DevicePool& d) {
                return d._device == device;
            }
        );
        size_t indx = gpu_it - _device.begin();
        _device[indx].free_vram = get_free_vram(_device[indx]._device);
        if (_device[indx].allocated_bytes > _device[indx].free_vram) {
            free_mem(indx);
        }
    }
    void Pool::free_mem(size_t indx_alloc) {
        auto& device = _device[indx_alloc];
        for (auto it = _segment.begin(); it != _segment.end();) {
            auto segment = *it;
            auto queue = segment->_queue;
            if (segment->is_free() && queue.get_device() == device._device) {
                auto head = segment->_head;

                while (head) {
                    for (auto it = _free_block.lower_bound(head->_bytes);
                        it != _free_block.upper_bound(head->_bytes);
                        it++) {
                            if (it->second == head) {
                                _free_block.erase(it);
                                break;
                            }
                        }

                    Block* next = head->next;
                    delete head;
                    head = next;
                }
                it = _segment.erase(it);
                sycl::free(segment->_base_ptr, queue);
                device.allocated_bytes -= segment->_total_bytes;
                delete segment;
            }
            else {
                ++it;
                continue;
            }
        }
    }
}
