#include "dispatcher/register_kernel.hpp"
#include "dispatcher/_table.hpp"
#include "tensor.hpp"
#include "kernel/kernel.hpp"






REGISTER_KERNEL(
    contiguous, 
    CPU,
    Int32,
    contiguous_CPU_Int32
);





REGISTER_KERNEL(
    contiguous, 
    CPU,
    Float32,
    contiguous_CPU_Float32
);





REGISTER_KERNEL(
    contiguous, 
    CPU,
    Float64,
    contiguous_CPU_Float64
);





REGISTER_KERNEL(
    contiguous, 
    CPU,
    Byte,
    contiguous_CPU_Byte
);





REGISTER_KERNEL(
    contiguous, 
    CPU,
    Char,
    contiguous_CPU_Char
);





REGISTER_KERNEL(
    contiguous, 
    CPU,
    Short,
    contiguous_CPU_Short
);





REGISTER_KERNEL(
    contiguous, 
    CPU,
    Long,
    contiguous_CPU_Long
);





REGISTER_KERNEL(
    contiguous, 
    CPU,
    ComplexFloat,
    contiguous_CPU_ComplexFloat
);





REGISTER_KERNEL(
    contiguous, 
    CPU,
    ComplexDouble,
    contiguous_CPU_ComplexDouble
);





REGISTER_KERNEL(
    contiguous, 
    CPU,
    Bool,
    contiguous_CPU_Bool
);







REGISTER_KERNEL(
    contiguous, 
    XPU,
    Int32,
    contiguous_XPU_Int32
);





REGISTER_KERNEL(
    contiguous, 
    XPU,
    Float32,
    contiguous_XPU_Float32
);





REGISTER_KERNEL(
    contiguous, 
    XPU,
    Float64,
    contiguous_XPU_Float64
);





REGISTER_KERNEL(
    contiguous, 
    XPU,
    Byte,
    contiguous_XPU_Byte
);





REGISTER_KERNEL(
    contiguous, 
    XPU,
    Char,
    contiguous_XPU_Char
);





REGISTER_KERNEL(
    contiguous, 
    XPU,
    Short,
    contiguous_XPU_Short
);





REGISTER_KERNEL(
    contiguous, 
    XPU,
    Long,
    contiguous_XPU_Long
);





REGISTER_KERNEL(
    contiguous, 
    XPU,
    ComplexFloat,
    contiguous_XPU_ComplexFloat
);





REGISTER_KERNEL(
    contiguous, 
    XPU,
    ComplexDouble,
    contiguous_XPU_ComplexDouble
);





REGISTER_KERNEL(
    contiguous, 
    XPU,
    Bool,
    contiguous_XPU_Bool
);









REGISTER_KERNEL(
    add, 
    CPU,
    Int32,
    add_CPU_Int32
);





REGISTER_KERNEL(
    add, 
    CPU,
    Float32,
    add_CPU_Float32
);





REGISTER_KERNEL(
    add, 
    CPU,
    Float64,
    add_CPU_Float64
);





REGISTER_KERNEL(
    add, 
    CPU,
    Byte,
    add_CPU_Byte
);





REGISTER_KERNEL(
    add, 
    CPU,
    Char,
    add_CPU_Char
);





REGISTER_KERNEL(
    add, 
    CPU,
    Short,
    add_CPU_Short
);





REGISTER_KERNEL(
    add, 
    CPU,
    Long,
    add_CPU_Long
);





REGISTER_KERNEL(
    add, 
    CPU,
    ComplexFloat,
    add_CPU_ComplexFloat
);





REGISTER_KERNEL(
    add, 
    CPU,
    ComplexDouble,
    add_CPU_ComplexDouble
);





REGISTER_KERNEL(
    add, 
    CPU,
    Bool,
    add_CPU_Bool
);







REGISTER_KERNEL(
    add, 
    XPU,
    Int32,
    add_XPU_Int32
);





REGISTER_KERNEL(
    add, 
    XPU,
    Float32,
    add_XPU_Float32
);





REGISTER_KERNEL(
    add, 
    XPU,
    Float64,
    add_XPU_Float64
);





REGISTER_KERNEL(
    add, 
    XPU,
    Byte,
    add_XPU_Byte
);





REGISTER_KERNEL(
    add, 
    XPU,
    Char,
    add_XPU_Char
);





REGISTER_KERNEL(
    add, 
    XPU,
    Short,
    add_XPU_Short
);





REGISTER_KERNEL(
    add, 
    XPU,
    Long,
    add_XPU_Long
);





REGISTER_KERNEL(
    add, 
    XPU,
    ComplexFloat,
    add_XPU_ComplexFloat
);





REGISTER_KERNEL(
    add, 
    XPU,
    ComplexDouble,
    add_XPU_ComplexDouble
);





REGISTER_KERNEL(
    add, 
    XPU,
    Bool,
    add_XPU_Bool
);









REGISTER_KERNEL(
    sub, 
    CPU,
    Int32,
    sub_CPU_Int32
);





REGISTER_KERNEL(
    sub, 
    CPU,
    Float32,
    sub_CPU_Float32
);





REGISTER_KERNEL(
    sub, 
    CPU,
    Float64,
    sub_CPU_Float64
);





REGISTER_KERNEL(
    sub, 
    CPU,
    Byte,
    sub_CPU_Byte
);





REGISTER_KERNEL(
    sub, 
    CPU,
    Char,
    sub_CPU_Char
);





REGISTER_KERNEL(
    sub, 
    CPU,
    Short,
    sub_CPU_Short
);





REGISTER_KERNEL(
    sub, 
    CPU,
    Long,
    sub_CPU_Long
);





REGISTER_KERNEL(
    sub, 
    CPU,
    ComplexFloat,
    sub_CPU_ComplexFloat
);





REGISTER_KERNEL(
    sub, 
    CPU,
    ComplexDouble,
    sub_CPU_ComplexDouble
);





REGISTER_KERNEL(
    sub, 
    CPU,
    Bool,
    sub_CPU_Bool
);







REGISTER_KERNEL(
    sub, 
    XPU,
    Int32,
    sub_XPU_Int32
);





REGISTER_KERNEL(
    sub, 
    XPU,
    Float32,
    sub_XPU_Float32
);





REGISTER_KERNEL(
    sub, 
    XPU,
    Float64,
    sub_XPU_Float64
);





REGISTER_KERNEL(
    sub, 
    XPU,
    Byte,
    sub_XPU_Byte
);





REGISTER_KERNEL(
    sub, 
    XPU,
    Char,
    sub_XPU_Char
);





REGISTER_KERNEL(
    sub, 
    XPU,
    Short,
    sub_XPU_Short
);





REGISTER_KERNEL(
    sub, 
    XPU,
    Long,
    sub_XPU_Long
);





REGISTER_KERNEL(
    sub, 
    XPU,
    ComplexFloat,
    sub_XPU_ComplexFloat
);





REGISTER_KERNEL(
    sub, 
    XPU,
    ComplexDouble,
    sub_XPU_ComplexDouble
);





REGISTER_KERNEL(
    sub, 
    XPU,
    Bool,
    sub_XPU_Bool
);









REGISTER_KERNEL(
    mul, 
    CPU,
    Int32,
    mul_CPU_Int32
);





REGISTER_KERNEL(
    mul, 
    CPU,
    Float32,
    mul_CPU_Float32
);





REGISTER_KERNEL(
    mul, 
    CPU,
    Float64,
    mul_CPU_Float64
);





REGISTER_KERNEL(
    mul, 
    CPU,
    Byte,
    mul_CPU_Byte
);





REGISTER_KERNEL(
    mul, 
    CPU,
    Char,
    mul_CPU_Char
);





REGISTER_KERNEL(
    mul, 
    CPU,
    Short,
    mul_CPU_Short
);





REGISTER_KERNEL(
    mul, 
    CPU,
    Long,
    mul_CPU_Long
);





REGISTER_KERNEL(
    mul, 
    CPU,
    ComplexFloat,
    mul_CPU_ComplexFloat
);





REGISTER_KERNEL(
    mul, 
    CPU,
    ComplexDouble,
    mul_CPU_ComplexDouble
);





REGISTER_KERNEL(
    mul, 
    CPU,
    Bool,
    mul_CPU_Bool
);







REGISTER_KERNEL(
    mul, 
    XPU,
    Int32,
    mul_XPU_Int32
);





REGISTER_KERNEL(
    mul, 
    XPU,
    Float32,
    mul_XPU_Float32
);





REGISTER_KERNEL(
    mul, 
    XPU,
    Float64,
    mul_XPU_Float64
);





REGISTER_KERNEL(
    mul, 
    XPU,
    Byte,
    mul_XPU_Byte
);





REGISTER_KERNEL(
    mul, 
    XPU,
    Char,
    mul_XPU_Char
);





REGISTER_KERNEL(
    mul, 
    XPU,
    Short,
    mul_XPU_Short
);





REGISTER_KERNEL(
    mul, 
    XPU,
    Long,
    mul_XPU_Long
);





REGISTER_KERNEL(
    mul, 
    XPU,
    ComplexFloat,
    mul_XPU_ComplexFloat
);





REGISTER_KERNEL(
    mul, 
    XPU,
    ComplexDouble,
    mul_XPU_ComplexDouble
);





REGISTER_KERNEL(
    mul, 
    XPU,
    Bool,
    mul_XPU_Bool
);









REGISTER_KERNEL(
    matmul, 
    CPU,
    Int32,
    matmul_CPU_Int32
);





REGISTER_KERNEL(
    matmul, 
    CPU,
    Float32,
    matmul_CPU_Float32
);





REGISTER_KERNEL(
    matmul, 
    CPU,
    Float64,
    matmul_CPU_Float64
);





REGISTER_KERNEL(
    matmul, 
    CPU,
    Byte,
    matmul_CPU_Byte
);





REGISTER_KERNEL(
    matmul, 
    CPU,
    Char,
    matmul_CPU_Char
);





REGISTER_KERNEL(
    matmul, 
    CPU,
    Short,
    matmul_CPU_Short
);





REGISTER_KERNEL(
    matmul, 
    CPU,
    Long,
    matmul_CPU_Long
);





REGISTER_KERNEL(
    matmul, 
    CPU,
    ComplexFloat,
    matmul_CPU_ComplexFloat
);





REGISTER_KERNEL(
    matmul, 
    CPU,
    ComplexDouble,
    matmul_CPU_ComplexDouble
);





REGISTER_KERNEL(
    matmul, 
    CPU,
    Bool,
    matmul_CPU_Bool
);







REGISTER_KERNEL(
    matmul, 
    XPU,
    Int32,
    matmul_XPU_Int32
);





REGISTER_KERNEL(
    matmul, 
    XPU,
    Float32,
    matmul_XPU_Float32
);





REGISTER_KERNEL(
    matmul, 
    XPU,
    Float64,
    matmul_XPU_Float64
);





REGISTER_KERNEL(
    matmul, 
    XPU,
    Byte,
    matmul_XPU_Byte
);





REGISTER_KERNEL(
    matmul, 
    XPU,
    Char,
    matmul_XPU_Char
);





REGISTER_KERNEL(
    matmul, 
    XPU,
    Short,
    matmul_XPU_Short
);





REGISTER_KERNEL(
    matmul, 
    XPU,
    Long,
    matmul_XPU_Long
);





REGISTER_KERNEL(
    matmul, 
    XPU,
    ComplexFloat,
    matmul_XPU_ComplexFloat
);





REGISTER_KERNEL(
    matmul, 
    XPU,
    ComplexDouble,
    matmul_XPU_ComplexDouble
);





REGISTER_KERNEL(
    matmul, 
    XPU,
    Bool,
    matmul_XPU_Bool
);









REGISTER_KERNEL(
    div, 
    CPU,
    Int32,
    div_CPU_Int32
);





REGISTER_KERNEL(
    div, 
    CPU,
    Float32,
    div_CPU_Float32
);





REGISTER_KERNEL(
    div, 
    CPU,
    Float64,
    div_CPU_Float64
);





REGISTER_KERNEL(
    div, 
    CPU,
    Byte,
    div_CPU_Byte
);





REGISTER_KERNEL(
    div, 
    CPU,
    Char,
    div_CPU_Char
);





REGISTER_KERNEL(
    div, 
    CPU,
    Short,
    div_CPU_Short
);





REGISTER_KERNEL(
    div, 
    CPU,
    Long,
    div_CPU_Long
);





REGISTER_KERNEL(
    div, 
    CPU,
    ComplexFloat,
    div_CPU_ComplexFloat
);





REGISTER_KERNEL(
    div, 
    CPU,
    ComplexDouble,
    div_CPU_ComplexDouble
);





REGISTER_KERNEL(
    div, 
    CPU,
    Bool,
    div_CPU_Bool
);







REGISTER_KERNEL(
    div, 
    XPU,
    Int32,
    div_XPU_Int32
);





REGISTER_KERNEL(
    div, 
    XPU,
    Float32,
    div_XPU_Float32
);





REGISTER_KERNEL(
    div, 
    XPU,
    Float64,
    div_XPU_Float64
);





REGISTER_KERNEL(
    div, 
    XPU,
    Byte,
    div_XPU_Byte
);





REGISTER_KERNEL(
    div, 
    XPU,
    Char,
    div_XPU_Char
);





REGISTER_KERNEL(
    div, 
    XPU,
    Short,
    div_XPU_Short
);





REGISTER_KERNEL(
    div, 
    XPU,
    Long,
    div_XPU_Long
);





REGISTER_KERNEL(
    div, 
    XPU,
    ComplexFloat,
    div_XPU_ComplexFloat
);





REGISTER_KERNEL(
    div, 
    XPU,
    ComplexDouble,
    div_XPU_ComplexDouble
);





REGISTER_KERNEL(
    div, 
    XPU,
    Bool,
    div_XPU_Bool
);









REGISTER_COPY_KERNEL(
    CPU,
    Int32,
    copy_CPU_Int32
);





REGISTER_COPY_KERNEL(
    CPU,
    Float32,
    copy_CPU_Float32
);





REGISTER_COPY_KERNEL(
    CPU,
    Float64,
    copy_CPU_Float64
);





REGISTER_COPY_KERNEL(
    CPU,
    Byte,
    copy_CPU_Byte
);





REGISTER_COPY_KERNEL(
    CPU,
    Char,
    copy_CPU_Char
);





REGISTER_COPY_KERNEL(
    CPU,
    Short,
    copy_CPU_Short
);





REGISTER_COPY_KERNEL(
    CPU,
    Long,
    copy_CPU_Long
);





REGISTER_COPY_KERNEL(
    CPU,
    ComplexFloat,
    copy_CPU_ComplexFloat
);





REGISTER_COPY_KERNEL(
    CPU,
    ComplexDouble,
    copy_CPU_ComplexDouble
);





REGISTER_COPY_KERNEL(
    CPU,
    Bool,
    copy_CPU_Bool
);







REGISTER_COPY_KERNEL(
    XPU,
    Int32,
    copy_XPU_Int32
);





REGISTER_COPY_KERNEL(
    XPU,
    Float32,
    copy_XPU_Float32
);





REGISTER_COPY_KERNEL(
    XPU,
    Float64,
    copy_XPU_Float64
);





REGISTER_COPY_KERNEL(
    XPU,
    Byte,
    copy_XPU_Byte
);





REGISTER_COPY_KERNEL(
    XPU,
    Char,
    copy_XPU_Char
);





REGISTER_COPY_KERNEL(
    XPU,
    Short,
    copy_XPU_Short
);





REGISTER_COPY_KERNEL(
    XPU,
    Long,
    copy_XPU_Long
);





REGISTER_COPY_KERNEL(
    XPU,
    ComplexFloat,
    copy_XPU_ComplexFloat
);





REGISTER_COPY_KERNEL(
    XPU,
    ComplexDouble,
    copy_XPU_ComplexDouble
);





REGISTER_COPY_KERNEL(
    XPU,
    Bool,
    copy_XPU_Bool
);









REGISTER_MEMCPY_KERNEL(
    cpymem,
    CPU,
    Int32,
    cpymem_CPU_Int32
);





REGISTER_MEMCPY_KERNEL(
    cpymem,
    CPU,
    Float32,
    cpymem_CPU_Float32
);





REGISTER_MEMCPY_KERNEL(
    cpymem,
    CPU,
    Float64,
    cpymem_CPU_Float64
);





REGISTER_MEMCPY_KERNEL(
    cpymem,
    CPU,
    Byte,
    cpymem_CPU_Byte
);





REGISTER_MEMCPY_KERNEL(
    cpymem,
    CPU,
    Char,
    cpymem_CPU_Char
);





REGISTER_MEMCPY_KERNEL(
    cpymem,
    CPU,
    Short,
    cpymem_CPU_Short
);





REGISTER_MEMCPY_KERNEL(
    cpymem,
    CPU,
    Long,
    cpymem_CPU_Long
);





REGISTER_MEMCPY_KERNEL(
    cpymem,
    CPU,
    ComplexFloat,
    cpymem_CPU_ComplexFloat
);





REGISTER_MEMCPY_KERNEL(
    cpymem,
    CPU,
    ComplexDouble,
    cpymem_CPU_ComplexDouble
);





REGISTER_MEMCPY_KERNEL(
    cpymem,
    CPU,
    Bool,
    cpymem_CPU_Bool
);







REGISTER_MEMCPY_KERNEL(
    cpymem,
    XPU,
    Int32,
    cpymem_XPU_Int32
);





REGISTER_MEMCPY_KERNEL(
    cpymem,
    XPU,
    Float32,
    cpymem_XPU_Float32
);





REGISTER_MEMCPY_KERNEL(
    cpymem,
    XPU,
    Float64,
    cpymem_XPU_Float64
);





REGISTER_MEMCPY_KERNEL(
    cpymem,
    XPU,
    Byte,
    cpymem_XPU_Byte
);





REGISTER_MEMCPY_KERNEL(
    cpymem,
    XPU,
    Char,
    cpymem_XPU_Char
);





REGISTER_MEMCPY_KERNEL(
    cpymem,
    XPU,
    Short,
    cpymem_XPU_Short
);





REGISTER_MEMCPY_KERNEL(
    cpymem,
    XPU,
    Long,
    cpymem_XPU_Long
);





REGISTER_MEMCPY_KERNEL(
    cpymem,
    XPU,
    ComplexFloat,
    cpymem_XPU_ComplexFloat
);





REGISTER_MEMCPY_KERNEL(
    cpymem,
    XPU,
    ComplexDouble,
    cpymem_XPU_ComplexDouble
);





REGISTER_MEMCPY_KERNEL(
    cpymem,
    XPU,
    Bool,
    cpymem_XPU_Bool
);









REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    CPU,
    Int32,
    copy_from_host_CPU_Int32
);





REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    CPU,
    Float32,
    copy_from_host_CPU_Float32
);





REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    CPU,
    Float64,
    copy_from_host_CPU_Float64
);





REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    CPU,
    Byte,
    copy_from_host_CPU_Byte
);





REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    CPU,
    Char,
    copy_from_host_CPU_Char
);





REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    CPU,
    Short,
    copy_from_host_CPU_Short
);





REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    CPU,
    Long,
    copy_from_host_CPU_Long
);





REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    CPU,
    ComplexFloat,
    copy_from_host_CPU_ComplexFloat
);





REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    CPU,
    ComplexDouble,
    copy_from_host_CPU_ComplexDouble
);





REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    CPU,
    Bool,
    copy_from_host_CPU_Bool
);







REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    XPU,
    Int32,
    copy_from_host_XPU_Int32
);





REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    XPU,
    Float32,
    copy_from_host_XPU_Float32
);





REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    XPU,
    Float64,
    copy_from_host_XPU_Float64
);





REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    XPU,
    Byte,
    copy_from_host_XPU_Byte
);





REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    XPU,
    Char,
    copy_from_host_XPU_Char
);





REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    XPU,
    Short,
    copy_from_host_XPU_Short
);





REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    XPU,
    Long,
    copy_from_host_XPU_Long
);





REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    XPU,
    ComplexFloat,
    copy_from_host_XPU_ComplexFloat
);





REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    XPU,
    ComplexDouble,
    copy_from_host_XPU_ComplexDouble
);





REGISTER_MEMCPY_KERNEL(
    copy_from_host,
    XPU,
    Bool,
    copy_from_host_XPU_Bool
);




