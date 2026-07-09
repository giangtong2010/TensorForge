#pragma once
#include <cstddef>

namespace dispatcher {
    enum class OP {
        
        contiguous,
        
        add,
        
        sub,
        
        mul,
        
        matmul,
        
        div,
        
        count = 6
    };
    enum class Backends {
        
        CPU,
        
        XPU,
        
        count = 2
    };
    enum class Types {
        
        Int32,
        
        Int64,
        
        Float32,
        
        Float64,
        
        Half,
        
        Byte,
        
        Char,
        
        Short,
        
        Long,
        
        ComplexHalf,
        
        ComplexFloat,
        
        ComplexDouble,
        
        Bool,
        
        count = 13
    };
}