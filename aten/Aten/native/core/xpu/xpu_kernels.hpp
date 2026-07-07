#pragma once
#include "tensor.hpp"






void contiguous_xpu(const at::Tensor&, const at::Tensor&, at::Tensor&);







void add_xpu(const at::Tensor&, const at::Tensor&, at::Tensor&);







void sub_xpu(const at::Tensor&, const at::Tensor&, at::Tensor&);







void mul_xpu(const at::Tensor&, const at::Tensor&, at::Tensor&);







void matmul_xpu(const at::Tensor&, const at::Tensor&, at::Tensor&);







void div_xpu(const at::Tensor&, const at::Tensor&, at::Tensor&);


