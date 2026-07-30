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
        
        cpymem,
        
        copy_from_host,
        
        count = 9
    };
}