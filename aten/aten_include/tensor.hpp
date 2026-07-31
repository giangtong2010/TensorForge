#pragma once
#include "_tensor.hpp"
#include <cstdint>
#include <cstddef>
#include <vector>
#include <iosfwd>

namespace at {
    class Tensor;

    namespace impl {
        // view ops func, define in _view_ops.cpp (FINSH)
        at::Tensor view(const at::Tensor&, const std::vector<int64_t>&);
        at::Tensor reshape(const at::Tensor, const std::vector<int64_t>);
        at::Tensor transpose(const at::Tensor&, const int64_t, const int64_t);
        at::Tensor permute(const at::Tensor&, const std::vector<int64_t>&);
        at::Tensor squeeze(const at::Tensor&, const size_t);
        at::Tensor unsqueeze(const at::Tensor&, const size_t);
        void squeeze_(at::Tensor&, const size_t);
        void unsqueeze_(at::Tensor&, const size_t);
        // reduction ops func, defin in _reduction_ops.cpp
        at::Tensor sum(const at::Tensor&);
        at::Tensor sum(const at::Tensor&, const int64_t);
        at::Tensor sum(const std::initializer_list<int64_t>&);
        at::Tensor mean(const at::Tensor&);
        at::Tensor max(const at::Tensor&);
        at::Tensor min(const at::Tensor&);
        // shape ops, define in _shape_ops.cpp
        at::Tensor flatten(const at::Tensor&);
        at::Tensor flatten(const at::Tensor&, int64_t, int64_t);
        at::Tensor expand(const at::Tensor&, const std::vector<int64_t>&);
        at::Tensor repeat(const at::Tensor&, const std::vector<int64_t>&);
        // memory layout, define in _memory_layout.cpp (FINISH)
        bool is_contiguous(const at::Tensor&) noexcept;
        at::Tensor contiguous(const at::Tensor&) noexcept;
        at::Tensor clone(const at::Tensor&) noexcept;
        // creation layout, define in _creation.cpp
        at::Tensor empty_like(const at::Tensor&) noexcept;
        at::Tensor empty(const std::vector<int64_t>& size, cpp20::Dtype dtype, cpp20::Device device);

        // arithmetic ops, defin in _arithmetic_ops.cpp (FINISH)
        at::Tensor add(const at::Tensor&, const at::Tensor&);
        at::Tensor sub(const at::Tensor&, const at::Tensor&);
        at::Tensor mul(const at::Tensor&, const at::Tensor&);
        at::Tensor div(const at::Tensor&, const at::Tensor&);
        at::Tensor matmul(const at::Tensor&, const at::Tensor&);
    }
}
