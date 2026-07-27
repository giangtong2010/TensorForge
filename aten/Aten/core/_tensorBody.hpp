#pragma once
#include "_tensor.hpp"
#include "tensor.hpp"

namespace at {
    Tensor Tensor::empty(const std::vector<int64_t>& size, cpp20::Dtype dtype, cpp20::Device device) {
        return empty(size, dtype, device);
    }
    Tensor Tensor::contiguous() const {
        return contiguous(*this);
    }
    Tensor Tensor::view(std::vector<int64_t>& index) const {
        return view(*this, index);
    }
}