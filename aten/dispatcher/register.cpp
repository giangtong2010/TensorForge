#include "dispatcher/register_kernel.hpp"
#include "dispatcher/_table.hpp"
#include "tensor.hpp"
#include "kernel/kernel.hpp"




REGISTER_KERNEL(
    contiguous, 
    cpu,
    contiguous_cpu
);



REGISTER_KERNEL(
    contiguous, 
    xpu,
    contiguous_xpu
);





REGISTER_KERNEL(
    add, 
    cpu,
    add_cpu
);



REGISTER_KERNEL(
    add, 
    xpu,
    add_xpu
);





REGISTER_KERNEL(
    sub, 
    cpu,
    sub_cpu
);



REGISTER_KERNEL(
    sub, 
    xpu,
    sub_xpu
);





REGISTER_KERNEL(
    mul, 
    cpu,
    mul_cpu
);



REGISTER_KERNEL(
    mul, 
    xpu,
    mul_xpu
);





REGISTER_KERNEL(
    matmul, 
    cpu,
    matmul_cpu
);



REGISTER_KERNEL(
    matmul, 
    xpu,
    matmul_xpu
);





REGISTER_KERNEL(
    div, 
    cpu,
    div_cpu
);



REGISTER_KERNEL(
    div, 
    xpu,
    div_xpu
);


