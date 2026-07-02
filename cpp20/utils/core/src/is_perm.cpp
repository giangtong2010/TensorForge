#include "is_perm.hpp"

namespace cpp20 {
    bool is_perm(const std::vector<int64_t>& perm, const std::vector<int64_t>& size) noexcept {
        std::vector<bool> visited(size.size(), false);
        size_t last_dim = size.size() - 1;
        if (perm.size() != size.size()) return false;

        for (size_t i = 0; i < perm.size(); i++) {
            int64_t v = perm[i];
            if (v < 0 || v > (int64_t) last_dim) return false;
            if (visited[v]) return false;
            visited[v] = true;
        }
        return true;
    }
}
