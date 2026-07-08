#include "xpu_kernels.hpp"
#include "utils.hpp"
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
#include <stdexcept>

namespace xpu {
    void kernel_contiguous_xpu(const at::Tensor& in, const at::Tensor&, at::Tensor& out) {
        if (in.get_device()._dev_type != cpp20::DeviceType::XPU) {
            throw std::runtime_error(
                "tensor argument for contiguous_xpu_kernel is not in xpu (intel gpu)"
            );
        }

        const uint8_t* old_data = in.data();
        const std::vector<int64_t> size = in.get_size();
        const std::vector<int64_t> old_stride = in.get_stride();
        const size_t numel = in.get_numel();
        const std::vector<int64_t> new_stride = cpp20::compute_strides(size);

        auto based_queues = based_queues::instance();
        sycl::queue& q = based_queues.get_queue(old_data);
        size_t qindx = based_queues.get_qindx(q);

        auto allocator = cpp20::get_allocator(in.get_device());
        cpp20::Data_ptr new_data_ptr = allocator.allocate(
            numel * cpp20::dtype_size(in.get_dtype()),
            qindx
        );

        sycl::device device = q.get_device();
        size_t elem_size = cpp20::dtype_size(in.get_dtype());
        size_t LOCAL = 256;
        size_t GLOBAL =
            ((numel + LOCAL - 1) / LOCAL) * LOCAL;

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
                uint8_t* new_data = new_data_ptr.get_data();
                for (size_t b = 0; b < elem_size; b++) {
                    new_data[k * elem_size + b] =
                        old_data[old_offset * elem_size + b];
                }
            }
        );

        auto storage_impl =
            cpp20::make_intrusive<at::StorageImpl>(
                new_data_ptr,
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
}
