#include "compute_strides.hpp"

namespace cpp20 {
    std::vector<int64_t> compute_strides(std::vector<int64_t>& size) noexcept {
        std::vector<int64_t> stride(size.size(), -1);
        size_t last_dim = size.size() - 1;
        stride[last_dim] = 1;

        for (size_t i = last_dim - 1; i-- > 0;) {
            stride[i] = stride[i + 1] * size[i + 1];
        }
        return stride;
    }
}
