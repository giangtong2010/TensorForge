#include "utils.hpp"
#include "Dtype.hpp"
#include "intrusive_ptr.hpp"
#include "_tensor_impl.hpp"
#include "_storage_impl.hpp"
#include "_tensor_iterator.hpp"
#include "config.hpp"
#include "cpu_memory_kn.hpp"

#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <tuple>
#include <utility>
#include <cstring>

namespace cpu {
    template <typename T>
    void kernel_cpymem_CPU(void* dst, const void* src, const cpp20::Dtype dtype, const size_t numel) {
        std::memcpy(
            dst,
            src,
            numel * sizeof(T)
        );
    }
    template <typename T>
    void kernel_copy_from_host_CPU(void* dst, const void* src, cpp20::Dtype dtype, const size_t numel) {
        const T* src_ptr = reinterpret_cast<const T*>(src);
        T* dst_ptr = reinterpret_cast<T*>(dst);

        dtype = cpp20::promote_dtype(
            dtype, cpp20::CPPTypeToDtype<T>::value
        );

        for (size_t i = 0; i < numel; i++) {
            dst_ptr[i] = src_ptr[i];
        }
    }
}
