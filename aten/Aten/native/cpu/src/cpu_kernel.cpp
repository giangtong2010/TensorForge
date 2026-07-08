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
    void kernel_contiguous_cpu(const at::Tensor& inp, const at::Tensor&, at::Tensor& out) {
        if (inp.get_device()._dev_type != cpp20::DeviceType::CPU) {
            throw std::runtime_error(
                "tensor argument for contiguous_cpu is not in cpu"
            );
        }

        const uint8_t* data = inp.data();
        const size_t numel = inp.get_numel();
        const std::vector<int64_t> size = inp.get_size();
        const std::vector<int64_t> old_stride = inp.get_stride();

        const std::vector<int64_t> new_stride = cpp20::compute_strides(size);
        auto& allocator = cpp20::get_allocator(inp.get_device());
        cpp20::Data_ptr new_data_ptr = allocator.allocate(
            numel * cpp20::dtype_size(inp.get_dtype())
        );

        parallel_for(
            0,
            numel,
            1024,
            [&](size_t begin, size_t end) {
                for (size_t k = begin; k < end; k++) {
                    size_t linear_indx = k;
                    std::vector<size_t> spatial_indx(size.size());

                    for (size_t i = 0; i < size.size(); i++) {
                        spatial_indx[i] = (linear_indx / new_stride[i]) % size[i];
                    }

                    size_t old_offset = cpp20::compute_offset(
                        inp.get_storage_offset(), 
                        spatial_indx, 
                        old_stride
                    );
                    size_t elem_size = cpp20::dtype_size(inp.get_dtype());
                    std::memcpy(
                        new_data_ptr.get_data() + k * elem_size,
                        data + old_offset * elem_size,
                        elem_size
                    );
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
}
