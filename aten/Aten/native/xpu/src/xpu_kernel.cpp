#include "xpu_kernels.hpp"
#include "utils.hpp"
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
    void kernel_contiguous_XPU(const at::Tensor& in, const at::Tensor&, at::Tensor& out) {
        if (in.get_device()._dev_type != cpp20::DeviceType::XPU) {
            throw std::runtime_error(
                "tensor argument for contiguous_xpu_kernel is not in xpu (intel gpu)"
            );
        }

        using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;

        const uint8_t* old_data = in.data();
        const std::vector<int64_t> size = in.get_size();
        const std::vector<int64_t> old_stride = in.get_stride();
        const size_t numel = in.get_numel();
        const std::vector<int64_t> new_stride = cpp20::compute_strides(size);

        auto based_queues = based_queues::instance();
        sycl::queue& q = based_queues.get_queue(old_data);
        size_t qindx = based_queues.get_qindx(q);

        auto& allocator = cpp20::get_allocator(in.get_device());
        cpp20::Data_ptr new_data_ptr = allocator.allocate(
            numel * cpp20::dtype_size(in.get_dtype()),
            qindx
        );
        uint8_t* new_data = new_data_ptr.get_data();

        sycl::device device = q.get_device();
        size_t elem_size = cpp20::dtype_size(in.get_dtype());
        size_t LOCAL = 256;
        size_t GLOBAL =
            ((numel + LOCAL - 1) / LOCAL) * LOCAL;

        auto* src = reinterpret_cast<const T*>(old_data);
        auto* dst = reinterpret_cast<T*>(new_data);

        q.parallel_for(
            sycl::nd_range<1>(GLOBAL, LOCAL),
            [&](sycl::nd_item<1> item) {
                size_t k = item.get_global_linear_id();

                if (k >= numel) {
                    return;
                }

                size_t linear_indx = k;
                size_t old_offset = in.get_storage_offset();

                for (size_t  i = 0; i < new_stride.size(); i++) {
                    auto indx = (linear_indx / new_stride[i]) % size[i];
                    old_offset += indx * old_stride[i];
                }
                
                dst[k] = src[old_offset];
            }
        );

        auto storage_impl =
            cpp20::make_intrusive<at::StorageImpl>(
                std::move(new_data_ptr),
                numel * cpp20::dtype_size(in.get_dtype()),
                in.get_device(),
                &allocator
            );
        at::Storage storage(storage_impl);
        auto tensor_impl =
            cpp20::make_intrusive<at::TensorImpl>(
                storage,
                size,
                in.get_dtype(),
                in.get_device(),
                0
            );
        
        out.get_impl() = tensor_impl;
    }

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

    template <typename T>
    void kernel_copy_XPU(at::Tensor& dst, const at::Tensor& src) {
        if (dst.get_numel() != src.get_numel())
            throw std::runtime_error("copy: tensor sizes do not match");
        if (!dst.is_contiguous() || !src.is_contiguous())
            throw std::runtime_error("copy: tensors must be contiguous");

        T* dst_ptr = reinterpret_cast<T*>(dst.data());
        const T* src_ptr = reinterpret_cast<T*>(src.data());

        sycl::queue& q_dst = based_queues::instance().get_queue(dst_ptr);
        sycl::queue& q_src = based_queues::instance().get_queue(src_ptr);

        assert(q_dst.get_context() == q_src.get_context());
        assert(q_dst.get_device() == q_src.get_device());

        q_dst.memcpy(
            dst_ptr,
            src_ptr,
            src.get_numel() * sizeof(T)
        ).wait();
    }

    template <typename T>
    void kernel_add_XPU(const at::TensorIterator& iter) {
        xpu_kernel<T>(
            iter, 
            [&](auto a, auto b) {return a + b;}
        );
    }
    template <typename T>
    void kernel_sub_XPU(const at::TensorIterator& iter) {
        xpu_kernel<T>(
            iter,
            [&](auto a, auto b) {return a - b;}
        );
    }
    template <typename T>
    void kernel_mul_XPU(const at::TensorIterator& iter) {
        xpu_kernel<T>(
            iter,
            [&](auto a, auto b) {return a * b;}
        );
    }
    template <typename T>
    void kernel_div_XPU(const at::TensorIterator& iter) {
        xpu_kernel<T>(
            iter,
            [&](auto a, auto b) {return a / b;}
        );
    }
}
