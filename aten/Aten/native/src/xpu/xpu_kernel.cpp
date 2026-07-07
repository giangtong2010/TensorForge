#include "xpu_kernels.hpp"
#include "utils.hpp"
#include "Allocator.hpp"
#include "Dtype.hpp"
#include "tensor.hpp"
#include "intrusive_ptr.hpp"
#include "_tensor_impl.hpp"
#include "_storage_impl.hpp"

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstddef>
#include <stdexcept>

namespace xpu {
    void kernel_contiguous_xpu(const at::Tensor& in, const at::Tensor, at::Tensor out) {
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
        auto allocator = cpp20::get_allocator(in.get_device());
        cpp20::Data_ptr new_data_ptr = allocator.allocate(
            numel * cpp20::dtype_size(in.get_dtype())
        );

        
    }
}