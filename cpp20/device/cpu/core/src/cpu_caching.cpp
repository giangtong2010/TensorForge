#include "cpu_caching.hpp"
#include "_windows_alloc_delete.hpp"
#include <new>
#include <algorithm>
#include <windows.h>
#include <malloc.h>
#include <stdexcept>
#include <cstdlib>

size_t get_free_ram() {
    #ifdef _WIN32
        return 
            windows_services::Windows_Services::get_available_ram();
    #endif
}

namespace cpu {
    // =============== BLOCK ====================
    void Block::mark_allocated() {
        if (!allocated) {
            allocated = true;
            _segment->active_block += 1;
            _segment->allocated_bytes += _bytes;
        }
    }
    void Block::mark_free() {
        if (allocated) {
            allocated = false;
            _segment->active_block -= 1;
            _segment->allocated_bytes -= _bytes;
        }
    }

    // ================== POOL ======================
    Block* Pool::find_free_block(size_t bytes) {
        if (!_free_block.empty()) {
            for (size_t i = 0; i < _free_block.size(); i++) {
                auto block = _free_block[i];
                if (!block->allocated && block->_bytes >= bytes) {
                    std::lock_guard<std::mutex> lock(_mutex);
                    _free_block.erase(_free_block.begin() + i);
                    auto allocated_block = split(bytes, block);
                    allocated_block->mark_allocated();
                    return allocated_block;
                }
            }
        }
        auto segment = allocate_segment(bytes);
        return split(bytes, segment->_head);
    }
    Block* Pool::split(size_t request_size, Block* block) {
        if (block->_bytes > request_size + kMinBlockSize) {
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
            if (block->next) 
                block->next->prev = new_block;
            block->next = new_block;

            _free_block.push_back(new_block);
        }
        block->mark_allocated();
        return block;
    }
    Block* Pool::merge(Block* this_block, Block* other_block) {
        if (!this_block || !other_block)
            return this_block ? this_block : other_block;

        if (this_block->next != other_block)
            return this_block;

        if (
            this_block->_ptr + this_block->_bytes == other_block->_ptr && 
            (!this_block->allocated && !other_block->allocated)
        ) {
            this_block->_bytes += other_block->_bytes;
            this_block->next = other_block->next;

            if (other_block->next)
                other_block->next->prev = this_block;
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _free_block.erase(
                    std::remove(_free_block.begin(), _free_block.end(), other_block),
                    _free_block.end()
                );
            }
            delete other_block;
            return this_block;
        }
        else {
            throw std::invalid_argument(
                "Two argument for this function do not continuos in memory"
            );
        }
    }

    Segment* Pool::allocate_segment(size_t bytes) {
        size_t alinged_bytes = 
            ((bytes + kCPUAligment - 1) / kCPUAligment) * kCPUAligment;
        #ifdef _WIN32
        void* ptr = 
                windows_services::Windows_Services::windows_allocate(
                    alinged_bytes, kCPUAligment
                );
        #endif
        if (!ptr) {
            free_mem();
            #ifdef _WIN32
            ptr = windows_services::Windows_Services::windows_allocate(
                alinged_bytes, kCPUAligment
            );
            #endif
        }
        try {
            Segment* segment = new Segment {
                static_cast<uint8_t*>(ptr),
                alinged_bytes,    
            };
            Block* head = new Block {
                static_cast<uint8_t*>(ptr),
                alinged_bytes,
            };
            head->_segment = segment;
            head->allocated = false;
            segment->_head = head;

            _segment.push_back(segment);
            _free_block.push_back(head);
            
            cached_bytes += segment->allocated_bytes;
            active_block += segment->active_block;
            if (cached_bytes - (128 * 1024 * 1024) > availabel_ram_bytes) {
                availabel_ram_bytes = get_free_ram();
            }
            return segment;
        } catch (...) {
            #ifdef _WIN32
            windows_services::Windows_Services::windows_free(ptr);
            #endif
            throw;
        }
    }

    void Pool::free_mem() {
        for (size_t i = 0; i < _free_block.size(); i++) {
            auto block = _free_block[i];
            auto segment = block->_segment;
            bool free = segment->is_free();

            auto head = segment->_head;
            if (free)
                #ifdef _WIN32
                windows_services::Windows_Services::windows_free(segment->_base_ptr);
                #endif
            while (head) {
                std::lock_guard<std::mutex> lock(_mutex);
                _free_block.erase(
                    std::remove(_free_block.begin(), _free_block.end(), head),
                    _free_block.end()
                );
                
                Block* next = head->next;
                delete head;
                head = next;
            }
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _segment.erase(
                    std::remove(_segment.begin(), _segment.end(), segment),
                    _segment.end()
                );
            }
            delete segment;
        }
    }
    void Pool::delete_block(Block* block) {
        if (!block) return;
        
        block->mark_free();
        if (block->prev && !block->prev->allocated)
            block = merge(block->prev, block);

        if (block->next && !block->next->allocated)
            block = merge(block, block->next);

        _free_block.push_back(block);
        availabel_ram_bytes = get_free_ram();
        if (cached_bytes > availabel_ram_bytes) {
            free_mem();
        }
    }
}
