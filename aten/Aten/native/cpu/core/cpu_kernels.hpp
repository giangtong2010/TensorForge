#pragma once
#include "tensor.hpp"





namespace cpu {
    void kernel_contiguous_CPU_Int32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_contiguous_CPU_Int32(a, b, c);
    return c;
};


namespace cpu {
    void kernel_contiguous_CPU_Int64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_Int64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_contiguous_CPU_Int64(a, b, c);
    return c;
};


namespace cpu {
    void kernel_contiguous_CPU_Float32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_contiguous_CPU_Float32(a, b, c);
    return c;
};


namespace cpu {
    void kernel_contiguous_CPU_Float64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_contiguous_CPU_Float64(a, b, c);
    return c;
};


namespace cpu {
    void kernel_contiguous_CPU_Half(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_contiguous_CPU_Half(a, b, c);
    return c;
};


namespace cpu {
    void kernel_contiguous_CPU_Byte(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_contiguous_CPU_Byte(a, b, c);
    return c;
};


namespace cpu {
    void kernel_contiguous_CPU_Char(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_contiguous_CPU_Char(a, b, c);
    return c;
};


namespace cpu {
    void kernel_contiguous_CPU_Short(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_contiguous_CPU_Short(a, b, c);
    return c;
};


namespace cpu {
    void kernel_contiguous_CPU_Long(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_contiguous_CPU_Long(a, b, c);
    return c;
};


namespace cpu {
    void kernel_contiguous_CPU_ComplexHalf(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_contiguous_CPU_ComplexHalf(a, b, c);
    return c;
};


namespace cpu {
    void kernel_contiguous_CPU_ComplexFloat(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_contiguous_CPU_ComplexFloat(a, b, c);
    return c;
};


namespace cpu {
    void kernel_contiguous_CPU_ComplexDouble(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_contiguous_CPU_ComplexDouble(a, b, c);
    return c;
};


namespace cpu {
    void kernel_contiguous_CPU_Bool(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_contiguous_CPU_Bool(a, b, c);
    return c;
};










namespace cpu {
    void kernel_add_CPU_Int32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_add_CPU_Int32(a, b, c);
    return c;
};


namespace cpu {
    void kernel_add_CPU_Int64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_Int64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_add_CPU_Int64(a, b, c);
    return c;
};


namespace cpu {
    void kernel_add_CPU_Float32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_add_CPU_Float32(a, b, c);
    return c;
};


namespace cpu {
    void kernel_add_CPU_Float64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_add_CPU_Float64(a, b, c);
    return c;
};


namespace cpu {
    void kernel_add_CPU_Half(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_add_CPU_Half(a, b, c);
    return c;
};


namespace cpu {
    void kernel_add_CPU_Byte(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_add_CPU_Byte(a, b, c);
    return c;
};


namespace cpu {
    void kernel_add_CPU_Char(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_add_CPU_Char(a, b, c);
    return c;
};


namespace cpu {
    void kernel_add_CPU_Short(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_add_CPU_Short(a, b, c);
    return c;
};


namespace cpu {
    void kernel_add_CPU_Long(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_add_CPU_Long(a, b, c);
    return c;
};


namespace cpu {
    void kernel_add_CPU_ComplexHalf(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_add_CPU_ComplexHalf(a, b, c);
    return c;
};


namespace cpu {
    void kernel_add_CPU_ComplexFloat(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_add_CPU_ComplexFloat(a, b, c);
    return c;
};


namespace cpu {
    void kernel_add_CPU_ComplexDouble(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_add_CPU_ComplexDouble(a, b, c);
    return c;
};


namespace cpu {
    void kernel_add_CPU_Bool(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_add_CPU_Bool(a, b, c);
    return c;
};










namespace cpu {
    void kernel_sub_CPU_Int32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_sub_CPU_Int32(a, b, c);
    return c;
};


namespace cpu {
    void kernel_sub_CPU_Int64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_Int64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_sub_CPU_Int64(a, b, c);
    return c;
};


namespace cpu {
    void kernel_sub_CPU_Float32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_sub_CPU_Float32(a, b, c);
    return c;
};


namespace cpu {
    void kernel_sub_CPU_Float64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_sub_CPU_Float64(a, b, c);
    return c;
};


namespace cpu {
    void kernel_sub_CPU_Half(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_sub_CPU_Half(a, b, c);
    return c;
};


namespace cpu {
    void kernel_sub_CPU_Byte(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_sub_CPU_Byte(a, b, c);
    return c;
};


namespace cpu {
    void kernel_sub_CPU_Char(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_sub_CPU_Char(a, b, c);
    return c;
};


namespace cpu {
    void kernel_sub_CPU_Short(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_sub_CPU_Short(a, b, c);
    return c;
};


namespace cpu {
    void kernel_sub_CPU_Long(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_sub_CPU_Long(a, b, c);
    return c;
};


namespace cpu {
    void kernel_sub_CPU_ComplexHalf(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_sub_CPU_ComplexHalf(a, b, c);
    return c;
};


namespace cpu {
    void kernel_sub_CPU_ComplexFloat(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_sub_CPU_ComplexFloat(a, b, c);
    return c;
};


namespace cpu {
    void kernel_sub_CPU_ComplexDouble(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_sub_CPU_ComplexDouble(a, b, c);
    return c;
};


namespace cpu {
    void kernel_sub_CPU_Bool(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_sub_CPU_Bool(a, b, c);
    return c;
};










namespace cpu {
    void kernel_mul_CPU_Int32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_mul_CPU_Int32(a, b, c);
    return c;
};


namespace cpu {
    void kernel_mul_CPU_Int64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_Int64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_mul_CPU_Int64(a, b, c);
    return c;
};


namespace cpu {
    void kernel_mul_CPU_Float32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_mul_CPU_Float32(a, b, c);
    return c;
};


namespace cpu {
    void kernel_mul_CPU_Float64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_mul_CPU_Float64(a, b, c);
    return c;
};


namespace cpu {
    void kernel_mul_CPU_Half(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_mul_CPU_Half(a, b, c);
    return c;
};


namespace cpu {
    void kernel_mul_CPU_Byte(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_mul_CPU_Byte(a, b, c);
    return c;
};


namespace cpu {
    void kernel_mul_CPU_Char(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_mul_CPU_Char(a, b, c);
    return c;
};


namespace cpu {
    void kernel_mul_CPU_Short(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_mul_CPU_Short(a, b, c);
    return c;
};


namespace cpu {
    void kernel_mul_CPU_Long(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_mul_CPU_Long(a, b, c);
    return c;
};


namespace cpu {
    void kernel_mul_CPU_ComplexHalf(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_mul_CPU_ComplexHalf(a, b, c);
    return c;
};


namespace cpu {
    void kernel_mul_CPU_ComplexFloat(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_mul_CPU_ComplexFloat(a, b, c);
    return c;
};


namespace cpu {
    void kernel_mul_CPU_ComplexDouble(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_mul_CPU_ComplexDouble(a, b, c);
    return c;
};


namespace cpu {
    void kernel_mul_CPU_Bool(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_mul_CPU_Bool(a, b, c);
    return c;
};










namespace cpu {
    void kernel_matmul_CPU_Int32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_matmul_CPU_Int32(a, b, c);
    return c;
};


namespace cpu {
    void kernel_matmul_CPU_Int64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_Int64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_matmul_CPU_Int64(a, b, c);
    return c;
};


namespace cpu {
    void kernel_matmul_CPU_Float32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_matmul_CPU_Float32(a, b, c);
    return c;
};


namespace cpu {
    void kernel_matmul_CPU_Float64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_matmul_CPU_Float64(a, b, c);
    return c;
};


namespace cpu {
    void kernel_matmul_CPU_Half(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_matmul_CPU_Half(a, b, c);
    return c;
};


namespace cpu {
    void kernel_matmul_CPU_Byte(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_matmul_CPU_Byte(a, b, c);
    return c;
};


namespace cpu {
    void kernel_matmul_CPU_Char(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_matmul_CPU_Char(a, b, c);
    return c;
};


namespace cpu {
    void kernel_matmul_CPU_Short(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_matmul_CPU_Short(a, b, c);
    return c;
};


namespace cpu {
    void kernel_matmul_CPU_Long(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_matmul_CPU_Long(a, b, c);
    return c;
};


namespace cpu {
    void kernel_matmul_CPU_ComplexHalf(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_matmul_CPU_ComplexHalf(a, b, c);
    return c;
};


namespace cpu {
    void kernel_matmul_CPU_ComplexFloat(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_matmul_CPU_ComplexFloat(a, b, c);
    return c;
};


namespace cpu {
    void kernel_matmul_CPU_ComplexDouble(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_matmul_CPU_ComplexDouble(a, b, c);
    return c;
};


namespace cpu {
    void kernel_matmul_CPU_Bool(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_matmul_CPU_Bool(a, b, c);
    return c;
};










namespace cpu {
    void kernel_div_CPU_Int32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_div_CPU_Int32(a, b, c);
    return c;
};


namespace cpu {
    void kernel_div_CPU_Int64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_Int64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_div_CPU_Int64(a, b, c);
    return c;
};


namespace cpu {
    void kernel_div_CPU_Float32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_div_CPU_Float32(a, b, c);
    return c;
};


namespace cpu {
    void kernel_div_CPU_Float64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_div_CPU_Float64(a, b, c);
    return c;
};


namespace cpu {
    void kernel_div_CPU_Half(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_div_CPU_Half(a, b, c);
    return c;
};


namespace cpu {
    void kernel_div_CPU_Byte(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_div_CPU_Byte(a, b, c);
    return c;
};


namespace cpu {
    void kernel_div_CPU_Char(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_div_CPU_Char(a, b, c);
    return c;
};


namespace cpu {
    void kernel_div_CPU_Short(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_div_CPU_Short(a, b, c);
    return c;
};


namespace cpu {
    void kernel_div_CPU_Long(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_div_CPU_Long(a, b, c);
    return c;
};


namespace cpu {
    void kernel_div_CPU_ComplexHalf(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_div_CPU_ComplexHalf(a, b, c);
    return c;
};


namespace cpu {
    void kernel_div_CPU_ComplexFloat(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_div_CPU_ComplexFloat(a, b, c);
    return c;
};


namespace cpu {
    void kernel_div_CPU_ComplexDouble(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_div_CPU_ComplexDouble(a, b, c);
    return c;
};


namespace cpu {
    void kernel_div_CPU_Bool(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    cpu::kernel_div_CPU_Bool(a, b, c);
    return c;
};






