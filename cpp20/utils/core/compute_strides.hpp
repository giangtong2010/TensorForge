#pragma once
#include <cstddef>
#include <vector>
#include <cstdint>

namespace cpp20 {
    std::vector<int64_t> compute_strides(const std::vector<int64_t>&) noexcept;
}