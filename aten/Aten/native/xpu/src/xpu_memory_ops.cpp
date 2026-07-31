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
    void kernel_contiguous_XPU(at::Tensor& dst, const at::Tensor& src) {
        if (src.get_device()._dev_type != cpp20::DeviceType::XPU) {
            throw std::runtime_error(
                "tensor argument for contiguous_xpu_kernel is not src xpu (intel gpu)"
            );
        }

        using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;

        const uint8_t* old_data = src.data();
        const std::vector<int64_t> size = src.get_size();
        const std::vector<int64_t> old_stride = src.get_stride();
        const size_t numel = src.get_numel();
        const std::vector<int64_t> new_stride = cpp20::compute_strides(size);

        auto based_queues = based_queues::instance();
        sycl::queue& q = based_queues.get_queue(old_data);
        size_t qindx = based_queues.get_qindx(q);

        auto& allocator = cpp20::get_allocator(src.get_device());
        cpp20::Data_ptr new_data_ptr = allocator.allocate(
            numel * cpp20::dtype_size(src.get_dtype()),
            qindx
        );
        uint8_t* new_data = new_data_ptr.get_data();

        sycl::device device = q.get_device();
        size_t elem_size = cpp20::dtype_size(src.get_dtype());
        size_t offset = src.get_storage_offset();
        size_t LOCAL = 256;
        size_t GLOBAL =
            ((numel + LOCAL - 1) / LOCAL) * LOCAL;

        auto* src_ptr = reinterpret_cast<const T*>(old_data);
        auto* dst_ptr = reinterpret_cast<T*>(new_data);

        int64_t* size_dev = xpu::utils::malloc_shared_copy<int64_t>(size, size.size(), q);
        int64_t* old_stride_dev = xpu::utils::malloc_shared_copy<int64_t>(old_stride, old_stride.size(), q);
        int64_t* new_stride_dev = xpu::utils::malloc_shared_copy<int64_t>(new_stride, new_stride.size(), q);

        size_t n = new_stride.size();

        q.parallel_for(
            sycl::nd_range<1>(GLOBAL, LOCAL),
            [=](sycl::nd_item<1> item) {
                size_t k = item.get_global_linear_id();

                if (k >= numel) {
                    return;
                }

                size_t linear_indx = k;
                size_t old_offset = offset;

                for (size_t  i = 0; i < n; i++) {
                    auto indx = (linear_indx / new_stride_dev[i]) % size_dev[i];
                    old_offset += indx * old_stride_dev[i];
                }
                
                dst_ptr[k] = src_ptr[old_offset];
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
    void kernel_copy_XPU(at::Tensor& dst, const at::Tensor& src) {
        if (dst.get_numel() != src.get_numel())
            throw std::runtime_error("copy: tensor sizes do not match");
        if (!dst.is_contiguous() || !src.is_contiguous())
            throw std::runtime_error("copy: tensors must be contiguous");

        T* dst_ptr = reinterpret_cast<T*>(dst.data());
        const T* src_ptr = reinterpret_cast<const T*>(src.data());

        sycl::queue& q_dst = based_queues::instance().get_queue(reinterpret_cast<uint8_t*>(dst_ptr));
        sycl::queue& q_src = based_queues::instance().get_queue(reinterpret_cast<const uint8_t*>(src_ptr));

        assert(q_dst.get_context() == q_src.get_context());
        assert(q_dst.get_device() == q_src.get_device());

        q_dst.memcpy(
            dst_ptr,
            src_ptr,
            src.get_numel() * sizeof(T)
        ).wait();
    }
}
