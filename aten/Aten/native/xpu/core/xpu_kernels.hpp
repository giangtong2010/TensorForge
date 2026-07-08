#pragma once
#include "tensor.hpp"






namespace xpu {
    void kernel_contiguous_xpu(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_xpu(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_contiguous_xpu(a, b, c);
    return c;
};







namespace xpu {
    void kernel_add_xpu(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_xpu(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_add_xpu(a, b, c);
    return c;
};







namespace xpu {
    void kernel_sub_xpu(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_xpu(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_sub_xpu(a, b, c);
    return c;
};







namespace xpu {
    void kernel_mul_xpu(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_xpu(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_mul_xpu(a, b, c);
    return c;
};







namespace xpu {
    void kernel_matmul_xpu(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_xpu(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_matmul_xpu(a, b, c);
    return c;
};







namespace xpu {
    void kernel_div_xpu(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_xpu(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_div_xpu(a, b, c);
    return c;
};


