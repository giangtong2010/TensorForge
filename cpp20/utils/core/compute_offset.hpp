#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

namespace cpp20 {
    SYCL_EXTERNAL size_t compute_offset(
        size_t storage_offset,
        const std::vector<int64_t>& size,
        const std::vector<int64_t>& stride
    ) noexcept;
    
    SYCL_EXTERNAL size_t compute_offset(
        size_t, 
        const std::vector<size_t>&, 
        const std::vector<int64_t>&
    ) noexcept;

    SYCL_EXTERNAL size_t compute_offset(
        size_t,
        size_t, 
        const std::vector<int64_t>&,
        const std::vector<int64_t>&
    ) noexcept;

    SYCL_EXTERNAL size_t compute_offset(
        size_t,
        size_t,
        const size_t,
        const int64_t*,
        const int64_t*
    ) noexcept;
}
