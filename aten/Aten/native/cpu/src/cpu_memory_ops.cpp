#include "utils.hpp"
#include "Dtype.hpp"
#include "intrusive_ptr.hpp"
#include "_tensor_impl.hpp"
#include "_storage_impl.hpp"
#include "cpu_memory_ops.hpp"
#include "config.hpp"
#include "tensor.hpp"

#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <tuple>
#include <utility>
#include <cstring>

namespace cpu {
    template <typename T>
    void kernel_contiguous_CPU(at::Tensor& dst, const at::Tensor& src) {
        if (src.get_device()._dev_type != cpp20::DeviceType::CPU) {
            throw std::runtime_error(
                "tensor argument for contiguous_cpu is not in cpu"
            );
        }

        const uint8_t* old_data = src.data();
        const size_t numel = src.get_numel();
        const std::vector<int64_t> size = src.get_size();
        const std::vector<int64_t> old_stride = src.get_stride();

        const std::vector<int64_t> new_stride = cpp20::compute_strides(size);
        auto& allocator = cpp20::get_allocator(src.get_device());
        cpp20::Data_ptr new_data_ptr = allocator.allocate(
            numel * cpp20::dtype_size(src.get_dtype())
        );

        auto* src_ptr = reinterpret_cast<const T*>(old_data);
        auto* dst_ptr = reinterpret_cast<T*>(new_data_ptr.get_data());

        parallel_for(
            0,
            numel,
            1024,
            [&](size_t begin, size_t end) {
                for (size_t k = begin; k < end; k++) {
                    size_t linear_indx = k;
                    size_t old_offset = src.get_storage_offset();

                    for (size_t i = 0; i < size.size(); i++) {
                        auto indx = (linear_indx / new_stride[i]) % size[i];
                        old_offset *= indx;
                    }

                    dst_ptr[k] = src_ptr[old_offset];
                }
            }
        );

        auto storage_impl =
            cpp20::make_intrusive<at::StorageImpl>(
                std::move(new_data_ptr),
                numel * cpp20::dtype_size(src.get_dtype()),
                src.get_device(),
                &allocator
            );
        at::Storage storage(storage_impl);
        auto tensor_impl = 
            cpp20::make_intrusive<at::TensorImpl>(
                storage,
                size,
                src.get_dtype(),
                src.get_device(),
                0
            );

        dst.get_impl() = tensor_impl;
    }

    template <typename T>
    void kernel_copy_CPU(at::Tensor& dst, const at::Tensor& src) {
        if (dst.get_numel() != src.get_numel())
            throw std::runtime_error("copy: tensor sizes do not match");
        if (!dst.is_contiguous() || !src.is_contiguous())
            throw std::runtime_error("copy: tensors must be contiguous");
            
        std::memcpy(
            dst.data(),
            src.data(),
            src.get_numel() * sizeof(T)
        );
    }
}
