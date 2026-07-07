#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>

namespace cpp20 {
    size_t compute_offset(
        size_t, 
        const std::vector<size_t>&, 
        const std::vector<int64_t>&
    ) noexcept;
}
