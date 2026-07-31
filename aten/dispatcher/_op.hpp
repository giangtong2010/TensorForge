#pragma once
#include <cstddef>

namespace dispatcher {
    enum class OP {
        
        add,
        
        sub,
        
        mul,
        
        div,
        
        pow,
        
        contiguous,
        
        copy,
        
        cpymem,
        
        copy_from_host,
        
        matmul,
        
        count = 10
    };
}