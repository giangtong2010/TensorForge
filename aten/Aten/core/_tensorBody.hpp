#pragma once
#include "_tensor.hpp"
#include "tensor.hpp"

namespace at {
// view ops func
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
    Tensor Tensor::transpose(int64_t dim1, int64_t dim2) const {
        return at::impl::transpose(*this, dim1, dim2);
    }
    Tensor Tensor::permute(const std::vector<int64_t>& perm) const {
        return at::impl::permute(*this, perm);
    }
    Tensor Tensor::squeeze(size_t indx) const {
        return at::impl::squeeze(*this, indx);
    }
    Tensor Tensor::unsqueeze(size_t indx) const {
        return at::impl::unsqueeze(*this, indx);
    }
    void Tensor::squeeze_(size_t indx) {
        at::impl::squeeze_(*this, indx);
    }
    void Tensor::unsqueeze_(size_t indx) {
        at::impl::unsqueeze_(*this, indx);
    }

// memory layout
    Tensor Tensor::contiguous() const {
        return at::impl::contiguous(*this);
    }
    Tensor Tensor::clone() const {
        return at::impl::clone(*this);
    }
    bool Tensor::is_contiguous() const {
        return at::impl::is_contiguous(*this);
    }

// arithmetic ops
    Tensor Tensor::add(const Tensor& other) const {
        return at::impl::add(*this, other);
    }
    Tensor Tensor::sub(const Tensor& other) const {
        return at::impl::sub(*this, other);
    }
    Tensor Tensor::mul(const Tensor& other) const {
        return at::impl::mul(*this, other);
    }
    Tensor Tensor::div(const Tensor& other) const {
        return at::impl::div(*this, other);
    }
    Tensor Tensor::matmul(const Tensor& other) const {
        return at::impl::matmul(*this, other);
    }
}
