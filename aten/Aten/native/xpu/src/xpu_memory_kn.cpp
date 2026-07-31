#include "utils.hpp"
#include "_util.hpp"
#include "Allocator.hpp"
#include "Dtype.hpp"
#include "tensor.hpp"
#include "xpu.hpp"
#include "intrusive_ptr.hpp"
#include "_tensor_impl.hpp"
#include "_storage_impl.hpp"
#include "config.hpp"

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstddef>
#include <array>
#include <utility>
#include <stdexcept>

namespace xpu {
    template <typename T>
    void kernel_memcpy_XPU(void* dst, const void* src, const size_t numel) {
        sycl::queue& q_dst = based_queues::instance().get_queue(reinterpret_cast<uint8_t*>(dst));
        sycl::queue& q_src = based_queues::instance().get_queue(reinterpret_cast<const uint8_t*>(src));

        assert(q_dst.get_context() == q_src.get_context());
        assert(q_dst.get_device() == q_src.get_device());

        q_dst.memcpy(dst, src, numel * sizeof(T));
    }
    template <typename T>
    void kernel_copy_from_host_CPU(void* dst, const void* src, cpp20::Dtype dtype, const size_t numel) {
        sycl::queue& q_dst = based_queues::instance().get_queue(reinterpret_cast<uint8_t*>(dst));
        sycl::queue& q_src = based_queues::instance().get_queue(reinterpret_cast<const uint8_t*>(src));

        assert(q_dst.get_context() == q_src.get_context());
        assert(q_dst.get_device() == q_src.get_device());

        const T* src_ptr = reinterpret_cast<const T*>(src);
        T* dst_ptr = reinterpret_cast<T*>(dst);

        dtype = cpp20::promote_dtype(
            dtype, cpp20::CPPTypeToDtype<T>::value
        );

        q_dst.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> id) {
            auto i = id[0];
            dst_ptr[i] = src_ptr[i];
        });
    }
}
