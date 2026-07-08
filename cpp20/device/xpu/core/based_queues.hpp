#pragma once
#include <map>
#include <sycl/sycl.hpp>
#include <cstdint>
#include <stdexcept>

namespace xpu {
    class based_queues {
        std::unordered_map<const uint8_t*, sycl::queue> _based_queues = {};
        std::unordered_map<sycl::queue, size_t> _queue_indx = {};
    
    public:
        static based_queues& instance() {
            static based_queues based_queues;
            return based_queues;
        }

        void add_queues_and_based(const uint8_t* ptr, const sycl::queue& queue) noexcept {
            _based_queues[ptr] = queue;
        }
        sycl::queue& get_queue(const uint8_t* ptr) {
            if (!_based_queues.contains(ptr)) {
                throw std::invalid_argument(
                    "argument for get_queue in class based_queues does not in map"
                );
            }
            return _based_queues[ptr];
        }

        void add_queue_and_indx(const sycl::queue& queue, size_t indx) noexcept {
            _queue_indx[queue] = indx;
        }
        size_t get_qindx(const sycl::queue& queue) {
            if (!_queue_indx.contains(queue)) {
                throw std::invalid_argument(
                    "argument for get_queue in class based_queues does not in map"
                );
            }
            return _queue_indx[queue];
        }

    protected:
        based_queues() = default;
    };
}
