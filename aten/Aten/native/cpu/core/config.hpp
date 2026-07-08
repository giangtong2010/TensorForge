#pragma once
#include <cstddef>

template <class function>
void parallel_for(
    size_t begin, 
    size_t end, 
    size_t grain_size,
    function&& func
);

#include "config.tpp"
