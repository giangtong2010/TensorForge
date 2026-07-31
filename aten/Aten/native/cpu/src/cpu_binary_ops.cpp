#include "utils.hpp"
#include "Dtype.hpp"
#include "_tensor_iterator.hpp"
#include "config.hpp"
#include "cpu_binary_ops.hpp"

#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <tuple>
#include <utility>
#include <cstring>
#include <cmath>

namespace cpu {
    template <typename T>
    void kernel_add_CPU(const at::TensorIterator& iter) {
        cpu_kernel<T>(
            iter,
            [&](auto a, auto b) {return a + b;}
        );
    }
    template <typename T>
    void kernel_sub_CPU(const at::TensorIterator& iter) {
        cpu_kernel<T>(
            iter,
            [&](auto a, auto b) {return a - b;}
        );
    }
    template <typename T>
    void kernel_div_CPU(const at::TensorIterator& iter) {
        cpu_kernel<T>(
            iter,
            [&](auto a, auto b) {return a / b;}
        );
    }
    template <typename T>
    void kernel_mul_CPU(const at::TensorIterator& iter) {
        cpu_kernel<T>(
            iter,
            [&](auto a, auto b) {return a * b;}
        );
    }
    template <typename T>
    void kernel_pow_CPU(const at::TensorIterator& iter) {
        cpu_kernel<T>(
            iter,
            [&](auto a, auto b) {return std::pow(a, b);}
        );
    }
}
