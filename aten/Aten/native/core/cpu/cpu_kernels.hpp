#pragma once
#include "tensor.hpp"



void contiguous_cpu(const at::Tensor&, const at::Tensor&, at::Tensor);







void add_cpu(const at::Tensor&, const at::Tensor&, at::Tensor);







void sub_cpu(const at::Tensor&, const at::Tensor&, at::Tensor);







void mul_cpu(const at::Tensor&, const at::Tensor&, at::Tensor);







void matmul_cpu(const at::Tensor&, const at::Tensor&, at::Tensor);







void div_cpu(const at::Tensor&, const at::Tensor&, at::Tensor);




