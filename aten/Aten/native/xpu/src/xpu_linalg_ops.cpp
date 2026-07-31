#include "utils.hpp"
#include "_util.hpp"
#include "Allocator.hpp"
#include "Dtype.hpp"
#include "tensor.hpp"
#include "xpu.hpp"
#include "intrusive_ptr.hpp"
#include "_tensor_impl.hpp"
#include "_storage_impl.hpp"

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstddef>
#include <array>
#include <utility>
#include <stdexcept>

namespace xpu {
    template <typename T>
    void kernel_matmul_XPU(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
        if (
            a.get_device()._dev_type != cpp20::DeviceType::XPU ||
            b.get_device()._dev_type != cpp20::DeviceType::XPU
        )
            throw std::invalid_argument("arg for kernel matmul was not in xpu (intel gpu)");
        
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

        auto& based_queue = based_queues::instance();
        sycl::queue& q = based_queue.get_queue(a.data());
        size_t qindx = based_queue.get_qindx(q);

        std::vector<int64_t> out_size = a_size;
        out_size[a_size.size() - 1] = b_size[b_size.size() - 1];

        auto& allocator = cpp20::get_allocator(a.get_device());
        cpp20::Data_ptr out_data_ptr = 
            allocator.allocate(
                cpp20::compute_numel(out_size) * cpp20::dtype_size(a.get_dtype()),
                qindx
            );

        size_t nums_work_item = a_size[a_size.size() - 2] * b_size[b_size.size() - 1];
        size_t n = a_size[a_size.size() - 1];
        size_t cols = b_size[b_size.size() - 1];

        const auto* a_data = reinterpret_cast<const T*>(a.data());
        const auto* b_data = reinterpret_cast<const T*>(b.data());
        auto* out_data = reinterpret_cast<T*>(out_data_ptr.get_data());

        q.parallel_for(sycl::range<1>(nums_work_item), [=](sycl::id<1> id) {
            size_t indx = id[0];
            size_t row = indx / cols;
            size_t col = indx % cols;

            T sum = 0;
            
            for(size_t k = 0; k < n; k++) {
                size_t row_offset = row * n + k;
                size_t col_offset = k * cols + col;

                sum += a_data[row_offset] * b_data[col_offset];
            }
            out_data[indx] = sum;
        });

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
