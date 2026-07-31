#include "cpu_linalg_ops.hpp"
#include "utils.hpp"
#include "Dtype.hpp"
#include "intrusive_ptr.hpp"
#include "_tensor_impl.hpp"
#include "_storage_impl.hpp"
#include "_tensor_iterator.hpp"
#include "config.hpp"

#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <tuple>
#include <utility>
#include <cstring>

namespace cpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
        if (
            a.get_device()._dev_type != cpp20::DeviceType::CPU ||
            b.get_device()._dev_type != cpp20::DeviceType::CPU
        )
            throw std::invalid_argument(
                "arg for add_cpu kernel is not in cpu"
            );

        auto a_size = a.get_size();
        auto b_size = b.get_size();
        if (a_size[a_size.size() - 1] != b_size[b_size.size() - 2])
            throw std::invalid_argument(
                "tensor must be have be shape like [m, n] x [n, q]"
            );
        
        try {
            cpp20::Dtype out_dtype = cpp20::promote_dtype(a.get_dtype(), b.get_dtype());
        } catch (...) {
            throw std::runtime_error(
                "Cannot coalesce dtype with dtype of a and b"
            );
        }

        std::vector<int64_t> out_size = a_size;
        out_size[a_size.size() - 1] = b_size[b_size.size() - 1];
        
        auto& allocator = cpp20::get_allocator(a.get_device());
        cpp20::Data_ptr out_data_ptr = allocator.allocate(
            cpp20::compute_numel(out_size) * cpp20::dtype_size(a.get_dtype())
        );

        size_t nums_work_item = a_size[a_size.size() - 2] * b_size[b_size.size() - 1];
        size_t n = a_size[a_size.size() - 1];
        size_t cols = b_size[b_size.size() -1];

        auto* a_data = reinterpret_cast<const T*>(a.data());
        auto* b_data = reinterpret_cast<const T*>(b.data());
        auto* out_data = reinterpret_cast<T*>(out_data_ptr.get_data());

        parallel_for(
            0,
            nums_work_item,
            1024,
            [&](size_t begin, size_t end) {
                for (size_t indx = begin; indx < end; indx++) {
                    size_t row = indx / cols;
                    size_t col = indx % cols;

                    T sum = 0;

                    for(size_t k = 0; k < n; k++) {
                        size_t rows_offset = row * n + k;
                        size_t col_offset = k * cols + col;

                        sum += a_data[rows_offset] * b_data[col_offset];
                    }
                    out_data[indx] = sum;
                }
            }
        );

        auto storage_impl =
            cpp20::make_intrusive<at::StorageImpl>(
                std::move(out_data_ptr),
                a.get_numel() * cpp20::dtype_size(a.get_dtype()),
                a.get_device(),
                &allocator
            );
        at::Storage storage(storage_impl);
        auto tensor_impl =
            cpp20::make_intrusive<at::TensorImpl>(
                storage,
                out_size,
                a.get_dtype(),
                a.get_device(),
                0
            );
        out.get_impl() = tensor_impl;
    }
}
