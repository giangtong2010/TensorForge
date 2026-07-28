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
        
        copy,
        
        count = 7
    };
}