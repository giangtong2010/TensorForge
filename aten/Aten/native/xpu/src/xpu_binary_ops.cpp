#include "utils.hpp"
#include "_util.hpp"
#include "Allocator.hpp"
#include "Dtype.hpp"
#include "tensor.hpp"
#include "xpu.hpp"
#include "config.hpp"

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstddef>
#include <array>
#include <utility>
#include <stdexcept>
#include <cmath>

namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::TensorIterator& iter) {
        xpu_kernel<T>(
            iter, 
            [&](auto a, auto b) {return a + b;}
        );
    }
    template <typename T>
    void kernel_sub_XPU(const at::TensorIterator& iter) {
        xpu_kernel<T>(
            iter,
            [&](auto a, auto b) {return a - b;}
        );
    }
    template <typename T>
    void kernel_mul_XPU(const at::TensorIterator& iter) {
        xpu_kernel<T>(
            iter,
            [&](auto a, auto b) {return a * b;}
        );
    }
    template <typename T>
    void kernel_div_XPU(const at::TensorIterator& iter) {
        xpu_kernel<T>(
            iter,
            [&](auto a, auto b) {return a / b;}
        );
    }
    template <typename T>
    void kernel_pow_XPU(const at::Tensor& iter) {
        xpu_kernel<T>(
            iter,
            [&](auto a, auto b) {return std::pow(a, b);}
        );
    }
}
