#pragma once
#include <sycl/sycl.hpp>

namespace xpu::utils {
    template <typename T>
    T* malloc_shared_copy(const std::vector<T>& vec, const size_t ndim, sycl::queue& q) {
        T* vec_ptr = sycl::malloc_shared<T>(ndim, q);
        q.memcpy(vec_ptr, vec.data(), ndim * sizeof(T)).wait();
        return vec_ptr;
    }
}
