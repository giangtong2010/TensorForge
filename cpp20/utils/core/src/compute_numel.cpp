#include "compute_numel.hpp"

namespace cpp20 {
    size_t compute_numel(const std::vector<int64_t>& size) noexcept {
        int64_t numel = 1;
        for (size_t i = 0; i < size.size(); i++) {
            numel *= size[i];
        }
        return (size_t) numel;
    }
}
