#include <vector>
#include <cstdint>
#include <cstddef>
#include <mutex>
size_t get_free_ram();

namespace cpu {
    struct Block {
        uint8_t* _ptr;
        size_t _bytes;
        bool allocated;

        Block* prev = nullptr;
        Block* next = nullptr;
        Segment* _segment = nullptr;

        bool is_allocated() {
            return allocated ? true : false;
        }

        void mark_allocated();
        void mark_free();
    };

    struct Segment {
        uint8_t* _base_ptr;
        size_t _total_bytes;
        Block* _head;

        size_t allocated_bytes = 0;
        int active_block = 0;

        bool is_free() {
            if (active_block == 0 && allocated_bytes == 0) {
                return true;
            }
            return false;
        }
    };

    class Pool {
        std::vector<Block*> _free_block = {};
        std::vector<Segment*> _segment = {};
        std::mutex _mutex;
        
        size_t availabel_ram_bytes = get_free_ram();
        size_t cached_bytes = 0;
        int active_block = 0;
        const size_t kMinBlockSize = 64;
        const size_t kCPUAligment = 64;

    public:
        Segment* allocate_segment(size_t bytes);
        Block* split(size_t request_size, Block* block);
        Block* merge(Block* this_block, Block* other_block);
        void delete_block(Block* block);
        void free_mem();
        Block* find_free_block(size_t bytes);
    };
}
