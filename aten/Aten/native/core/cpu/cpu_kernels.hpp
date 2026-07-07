#pragma once
#include "tensor.hpp"




namespace cpu {
    void kernel_contiguous_cpu(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_cpu(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_contiguous_cpu(a, b, c);
    return c;
};







namespace cpu {
    void kernel_add_cpu(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_cpu(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_add_cpu(a, b, c);
    return c;
};







namespace cpu {
    void kernel_sub_cpu(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_cpu(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_sub_cpu(a, b, c);
    return c;
};







namespace cpu {
    void kernel_mul_cpu(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_cpu(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_mul_cpu(a, b, c);
    return c;
};







namespace cpu {
    void kernel_matmul_cpu(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_cpu(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_matmul_cpu(a, b, c);
    return c;
};







namespace cpu {
    void kernel_div_cpu(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_cpu(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_div_cpu(a, b, c);
    return c;
};




