#pragma once
#include <cstddef>
#include <stdexcept>
#include <cstdint>
#include "_Dtype.hpp"

namespace cpp20 {
    inline size_t dtype_size(Dtype dtype) {
        try {
            return dtype_size(dtype);
        }
        catch (const std::exception& e) {
            throw e;
        }
    }
}
