#pragma once
#include <cstddef>

template <class function>
void parallel_for(
    size_t begin, 
    size_t end, 
    size_t grain_size,
    function&& func
);

template <typename scalar_t, class Func>
void cpu_kernel(at::TensorIterator& iter, Func&& op);

#include "config.tpp"
