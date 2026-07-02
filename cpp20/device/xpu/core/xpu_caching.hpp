#include <sycl/sycl.hpp>
#include <cstdint>
#include <vector>
#include <cstddef>
#include <mutex>

namespace xpu {
    struct Block {
        uint8_t* _ptr = nullptr;
        size_t _bytes = 0;
        bool allocated;

        Block* prev;
        Block* next;
        Segment* _segment;

        bool is_allocated() {
            return allocated ? true : false;
        }

        void mark_allocated();
        void mark_free();
    };

    struct Segment {
        sycl::queue _queue;
        uint8_t* _base_ptr;
        size_t _total_bytes;
        Block* _head;

        size_t allocated_bytes = 0;
        size_t active_block = 0;

        bool is_free() {
            if (allocated_bytes == 0 && active_block == 0) {
                return true;
            }
            return false;
        }
    };

    class Pool {
        std::vector<Block*> _free_block = {};
        std::vector<Segment*> _segment = {};
        std::vector<sycl::queue> _queues = {};
        std::mutex _mutex;

        size_t indx_alloc = 0;

        size_t available_ram = 0;
        size_t active_bytes = 0;
        int active_block = 0;
        const size_t kMinBlockSize = 64;
        const size_t kMinXPUAlignment = 64;

    public:
        Segment* allocate_segment(size_t nbytes);
        Block* split(size_t request_size, Block* block);
        Block* merge(Block* a, Block* b);
        Block* find_free_block(size_t request_size);
        void delete_block(Block* block);
        void free_mem();

        Pool();
        ~Pool();
    };
}
