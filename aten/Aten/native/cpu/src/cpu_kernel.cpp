#include "cpu_kernels.hpp"
#include "utils.hpp"
#include "Allocator.hpp"
#include "Dtype.hpp"
#include "intrusive_ptr.hpp"
#include "_tensor_impl.hpp"
#include "_storage_impl.hpp"
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
    void kernel_contiguous_CPU(const at::Tensor& inp, const at::Tensor&, at::Tensor& out) {
        if (inp.get_device()._dev_type != cpp20::DeviceType::CPU) {
            throw std::runtime_error(
                "tensor argument for contiguous_cpu is not in cpu"
            );
        }

        const uint8_t* old_data = inp.data();
        const size_t numel = inp.get_numel();
        const std::vector<int64_t> size = inp.get_size();
        const std::vector<int64_t> old_stride = inp.get_stride();

        const std::vector<int64_t> new_stride = cpp20::compute_strides(size);
        auto& allocator = cpp20::get_allocator(inp.get_device());
        cpp20::Data_ptr new_data_ptr = allocator.allocate(
            numel * cpp20::dtype_size(inp.get_dtype())
        );

        auto* src = reinterpret_cast<const T*>(old_data);
        auto* dst = reinterpret_cast<T*>(new_data_ptr.get_data());

        parallel_for(
            0,
            numel,
            1024,
            [&](size_t begin, size_t end) {
                for (size_t k = begin; k < end; k++) {
                    size_t linear_indx = k;
                    size_t old_offset = inp.get_storage_offset();

                    for (size_t i = 0; i < size.size(); i++) {
                        auto indx = (linear_indx / new_stride[i]) % size[i];
                        old_offset *= indx;
                    }

                    dst[k] = src[old_offset];
                }
            }
        );

        auto storage_impl =
            cpp20::make_intrusive<at::StorageImpl>(
                std::move(new_data_ptr),
                numel * cpp20::dtype_size(inp.get_dtype()),
                inp.get_device(),
                &allocator
            );
        at::Storage storage(storage_impl);
        auto tensor_impl = 
            cpp20::make_intrusive<at::TensorImpl>(
                storage,
                size,
                inp.get_dtype(),
                inp.get_device(),
                0
            );

        out.get_impl() = tensor_impl;
    }

    template <typename T>
    void kernel_add_CPU(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
        if (
            a.get_device()._dev_type != cpp20::DeviceType::CPU ||
            b.get_device()._dev_type != cpp20::DeviceType::CPU
        )
            throw std::invalid_argument(
                "arg for add_cpu kernel is not in cpu"
            );
        if (a.get_size() != b.get_size())
            throw std::invalid_argument(
                "tensor must be have the same size to be add"
            );
        if (a.get_dtype() != b.get_dtype())
            throw std::runtime_error(
                "tensor must be have the same dtype to be added"
            );

        auto& allocator = cpp20::get_allocator(a.get_device());
        cpp20::Data_ptr new_data_ptr = allocator.allocate(
            a.get_numel() * cpp20::dtype_size(a.get_dtype())
        );

        auto* a_data = reinterpret_cast<const T*>(a.data());
        auto* b_data = reinterpret_cast<T*>(b.data());
        auto* out_data = reinterpret_cast<T*>(new_data_ptr.get_data());

        parallel_for(
            0,
            a.get_numel(),
            1024,
            [&](size_t begin, size_t end) {
                for (size_t k = begin; k < end; k++) {
                    size_t linear_indx = k;
                    size_t a_offset = cpp20::compute_offset(
                        a.get_storage_offset(),
                        linear_indx,
                        a.get_stride(),
                        a.get_size()
                    );
                    size_t b_offset = cpp20::compute_offset(
                        b.get_storage_offset(),
                        linear_indx,
                        b.get_stride(),
                        b.get_size()
                    );

                    out_data[k] = a_data[a_offset] + b_data[b_offset];
                }
            }
        );

        auto storage_impl =
            cpp20::make_intrusive<at::StorageImpl>(
                std::move(new_data_ptr),
                a.get_numel() * cpp20::dtype_size(a.get_dtype()),
                a.get_device(),
                &allocator
            );
        at::Storage storage(storage_impl);
        auto tensor_impl =
            cpp20::make_intrusive<at::TensorImpl>(
                storage,
                a.get_size(),
                a.get_dtype(),
                a.get_device(),
                0
            );
        out.get_impl() = tensor_impl;
    }

    template <typename T>
    void kernel_sub_CPU(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
        if (
            a.get_device()._dev_type != cpp20::DeviceType::CPU ||
            b.get_device()._dev_type != cpp20::DeviceType::CPU
        )
            throw std::invalid_argument(
                "arg for add_cpu kernel is not in cpu"
            );
        if (a.get_size() != b.get_size())
            throw std::invalid_argument(
                "tensor must be have the same size to be subed"
            );
        if (a.get_dtype() != b.get_dtype())
            throw std::runtime_error(
                "tensor must be have the same dtype to be subed"
            );

        auto& allocator = cpp20::get_allocator(a.get_device());
        cpp20::Data_ptr new_data_ptr = allocator.allocate(
            a.get_numel() * cpp20::dtype_size(a.get_dtype())
        );

        auto* a_data = reinterpret_cast<const T*>(a.data());
        auto* b_data = reinterpret_cast<const T*>(b.data());
        auto* out_data = reinterpret_cast<const T*>(new_data_ptr.get_data());

        parallel_for(
            0,
            a.get_numel(),
            [&](size_t begin, size_t end) {
                for(size_t linear_indx = begin; linear_indx < end; linear_indx++) {
                    size_t a_offset = cpp20::compute_offset(
                        linear_indx,
                        a.get_storage_offset(),
                        a.get_stride(),
                        a.get_size()
                    );
                    size_t b_offset = cpp20::compute_offset(
                        linear_indx, 
                        b.get_storage_offset(),
                        b.get_stride(),
                        b.get_size()
                    );

                    out_data[linear_indx] = a_data[a_offset] - b_data[b_offset];
                }
            }
        );

        auto storage_impl =
            cpp20::make_intrusive<at::StorageImpl>(
                std::move(new_data_ptr),
                a.get_numel() * cpp20::dtype_size(a.get_dtype()),
                a.get_device(),
                &allocator
            );
        at::Storage storage(storage_impl);
        auto tensor_impl =
            cpp20::make_intrusive<at::TensorImpl>(
                storage,
                a.get_size(),
                a.get_dtype(),
                a.get_device(),
                0
            );
        out.get_impl() = tensor_impl;
    }

    template <typename T>
    void kernel_mul_CPU(const at::Tensor& scalar, const at::Tensor& tensor, at::Tensor& out) {
        if (
            scalar.get_device()._dev_type != cpp20::DeviceType::CPU ||
            tensor.get_device()._dev_type != cpp20::DeviceType::CPU
        )
            throw std::invalid_argument(
                "arg for add_cpu kernel is not in cpu"
            );
        if (!scalar.get_size().empty())
            throw std::invalid_argument(
                "the first argument for mul must be scalar tensor"
            );
        if (scalar.get_dtype() != scalar.get_dtype())
            throw std::runtime_error(
                "tensor must be have the same dtype to be subed"
            );

        auto& allocator = cpp20::get_allocator(tensor.get_device());
        auto out_data_ptr = allocator.allocate(
            tensor.get_numel() * cpp20::dtype_size(tensor.get_dtype())
        );

        auto* scalar_data = reinterpret_cast<const T*>(scalar.data());
        auto* tensor_data = reinterpret_cast<const T*>(tensor.data());
        auto* out_data = reinterpret_cast<T*>(out_data_ptr.get_data());
        size_t tensor_numel = tensor.get_numel();

        parallel_for(
            0,
            tensor_numel,
            1024,
            [&](size_t begin, size_t end) {
                for (size_t k = begin; k < end; k++) {
                    size_t tensor_offset = cpp20::compute_offset(
                        tensor.get_storage_offset(),
                        k,
                        tensor.get_stride(),
                        tensor.get_size()
                    );
                    out_data[k] = scalar_data * tensor_data[tensor_offset];
                }
            }
        );

        auto storage_impl = 
            cpp20::make_intrusive<at::StorageImpl>(
                std::move(out_data_ptr),
                tensor.get_numel() * cpp20::dtype_size(tensor.get_dtype()),
                tensor.get_device(),
                &allocator
            );
        at::Storage storage(storage_impl);
        auto tensor_impl = 
            cpp20::make_intrusive<at::TensorImpl>(
                storage,
                tensor.get_size(),
                tensor.get_dtype(),
                tensor.get_device(),
                0
            );
        out.get_impl() = tensor_impl;
    }

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
        if (a.get_dtype() != b.get_dtype())
            throw std::runtime_error(
                "tensor must be have the same dtype to be matmul"
            );

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
