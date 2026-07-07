#pragma once
#include <cstddef>

template <class func>
void parallel_for(
    size_t begin, 
    size_t end, 
    size_t grain_size = 1024,
    func&& func
);

#include "config.tpp"
