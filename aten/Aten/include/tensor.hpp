#pragma once
#include "_tensor.hpp"
#include "_tensorBody.hpp"
#include <cstdint>
#include <cstddef>
#include <vector>

namespace at {
    // view ops func, define in _view_ops.cpp
    Tensor view(const Tensor&, const std::vector<int64_t>&);
    Tensor reshape(const Tensor, const std::vector<int64_t>);
    Tensor transpose(const Tensor&, const size_t, const size_t);
    Tensor permute(const Tensor&, const std::vector<int64_t>&);
    Tensor squeeze(const Tensor&, const size_t);
    Tensor unsqueeze(const Tensor&, const size_t);
    void squeeze_(Tensor&, const size_t);
    void unsqueeze_(Tensor&, const size_t);
    // reduction ops func, defin in _reduction_ops.cpp
    Tensor sum(const Tensor&);
    Tensor sum(const Tensor&, const int64_t);
    Tensor sum(const std::initializer_list<int64_t>&);
    Tensor mean(const Tensor&);
    Tensor max(const Tensor&);
    Tensor min(const Tensor&);
    // shape ops, define in _shape_ops.cpp
    Tensor flatten(const Tensor&);
    Tensor flatten(const Tensor&, int64_t, int64_t);
    Tensor expand(const Tensor&, const std::vector<int64_t>&);
    Tensor repeat(const Tensor&, const std::vector<int64_t>&);
    // memory layout, define in _memory_layout.cpp
    bool is_contiguous(const Tensor&) noexcept;
    Tensor contiguous(const Tensor&) noexcept;
    Tensor clone(const Tensor&) noexcept;
    Tensor detach(const Tensor&) noexcept;
}
