#pragma once
#include "_tensor.hpp"
#include "tensor.hpp"

namespace at {
    static_assert(true);
    Tensor Tensor::empty(const std::vector<int64_t>& size, cpp20::Dtype dtype, cpp20::Device device) {
        return at::impl::empty(size, dtype, device);
    }
    Tensor Tensor::contiguous() const {
        return at::impl::contiguous(*this);
    }
    Tensor Tensor::view(const std::vector<int64_t>& index) const {
        return at::impl::view(*this, index);
    }
    Tensor Tensor::reshape(const std::vector<int64_t>& index) const {
        return at::impl::reshape(*this, index);
    }
}
