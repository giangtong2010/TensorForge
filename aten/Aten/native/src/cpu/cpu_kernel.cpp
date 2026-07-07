#include "cpu_kernels.hpp"
#include <cstdint>
#include <cstddef>
#include <malloc.h>
#include <vector>
#include <ranges>
#include <new>

void contiguous_cpu(const Tensor& in, const Tensor& inp, Tensor& out) {
    std::vector<size_t> linear_indx(in.get_impl())
}
