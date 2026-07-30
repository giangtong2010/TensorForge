#include "compute_strides.hpp"
#include <vector>

namespace cpp20 {
    std::vector<int64_t> compute_strides(const std::vector<int64_t>& size) noexcept {
        if (size.empty())
            throw;

        std::vector<int64_t> stride(size.size());
        stride.back() = 1;

        for (int i = static_cast<int>(size.size()) - 2; i >=0; i--) {
            stride[i] = stride[i + 1] * size[i + 1];
        }
        return stride;
    }
}
