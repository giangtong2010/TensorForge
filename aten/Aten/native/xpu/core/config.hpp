#pragma once
#include "_tensor_iterator.hpp"

template <typename scalar_t, class Func>
void xpu_kernel(const at::TensorIterator& iter, Func&& op);

#include "config.tpp"
