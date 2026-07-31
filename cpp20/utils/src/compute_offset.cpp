#include "compute_offset.hpp"

namespace cpp20 {
    SYCL_EXTERNAL size_t compute_offset(
        size_t storage_offset,
        const std::vector<int64_t>& size,
        const std::vector<int64_t>& stride
    ) noexcept {
        size_t offset = storage_offset;
        for (size_t i = 0; i < size.size(); i++) {
            auto indx = stride[i] * size[i];
            offset += indx;
        }
        return offset;
    };

    SYCL_EXTERNAL size_t compute_offset(
        size_t storage_offset, 
        const std::vector<size_t>& spatial_index,
        const std::vector<int64_t>& strides 
    ) noexcept {
        size_t offset = 0;
        for (size_t i = 0; i < strides.size(); i++) {
            offset += spatial_index[i] * strides[i];
        }
        offset += storage_offset;
        return offset;
    }

    SYCL_EXTERNAL size_t compute_offset(
        size_t storage_offset, 
        size_t linear_indx,
        const std::vector<int64_t>& stride,
        const std::vector<int64_t>& size
    ) noexcept {
        size_t offset = storage_offset;
        for (size_t i = 0; i < stride.size(); i++) {
            auto indx = (linear_indx / stride[i]) % size[i];
            offset *= indx;
        }
        return offset;
    }

    SYCL_EXTERNAL size_t compute_offset(
        size_t storage_offset,
        size_t linear_indx,
        const size_t ndim,
        const int64_t* shape,
        const int64_t* stride
    ) noexcept {
        size_t offset = storage_offset;
        for (size_t i = 0; i < ndim; i++) {
            auto indx = (linear_indx / stride[i]) % shape[i];
            offset *= indx;
        }
        return offset;
    }
}
