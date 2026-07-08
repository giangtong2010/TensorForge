#include "compute_offset.hpp"

namespace cpp20 {
    size_t compute_offset(
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
}
