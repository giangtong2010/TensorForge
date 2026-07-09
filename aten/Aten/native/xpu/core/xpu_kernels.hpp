#pragma once
#include "tensor.hpp"







namespace xpu {
    void kernel_contiguous_XPU_Int32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_contiguous_XPU_Int32(a, b, c);
    return c;
};


namespace xpu {
    void kernel_contiguous_XPU_Int64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_Int64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_contiguous_XPU_Int64(a, b, c);
    return c;
};


namespace xpu {
    void kernel_contiguous_XPU_Float32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_contiguous_XPU_Float32(a, b, c);
    return c;
};


namespace xpu {
    void kernel_contiguous_XPU_Float64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_contiguous_XPU_Float64(a, b, c);
    return c;
};


namespace xpu {
    void kernel_contiguous_XPU_Half(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_contiguous_XPU_Half(a, b, c);
    return c;
};


namespace xpu {
    void kernel_contiguous_XPU_Byte(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_contiguous_XPU_Byte(a, b, c);
    return c;
};


namespace xpu {
    void kernel_contiguous_XPU_Char(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_contiguous_XPU_Char(a, b, c);
    return c;
};


namespace xpu {
    void kernel_contiguous_XPU_Short(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_contiguous_XPU_Short(a, b, c);
    return c;
};


namespace xpu {
    void kernel_contiguous_XPU_Long(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_contiguous_XPU_Long(a, b, c);
    return c;
};


namespace xpu {
    void kernel_contiguous_XPU_ComplexHalf(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_contiguous_XPU_ComplexHalf(a, b, c);
    return c;
};


namespace xpu {
    void kernel_contiguous_XPU_ComplexFloat(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_contiguous_XPU_ComplexFloat(a, b, c);
    return c;
};


namespace xpu {
    void kernel_contiguous_XPU_ComplexDouble(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_contiguous_XPU_ComplexDouble(a, b, c);
    return c;
};


namespace xpu {
    void kernel_contiguous_XPU_Bool(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_contiguous_XPU_Bool(a, b, c);
    return c;
};










namespace xpu {
    void kernel_add_XPU_Int32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_add_XPU_Int32(a, b, c);
    return c;
};


namespace xpu {
    void kernel_add_XPU_Int64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_Int64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_add_XPU_Int64(a, b, c);
    return c;
};


namespace xpu {
    void kernel_add_XPU_Float32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_add_XPU_Float32(a, b, c);
    return c;
};


namespace xpu {
    void kernel_add_XPU_Float64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_add_XPU_Float64(a, b, c);
    return c;
};


namespace xpu {
    void kernel_add_XPU_Half(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_add_XPU_Half(a, b, c);
    return c;
};


namespace xpu {
    void kernel_add_XPU_Byte(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_add_XPU_Byte(a, b, c);
    return c;
};


namespace xpu {
    void kernel_add_XPU_Char(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_add_XPU_Char(a, b, c);
    return c;
};


namespace xpu {
    void kernel_add_XPU_Short(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_add_XPU_Short(a, b, c);
    return c;
};


namespace xpu {
    void kernel_add_XPU_Long(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_add_XPU_Long(a, b, c);
    return c;
};


namespace xpu {
    void kernel_add_XPU_ComplexHalf(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_add_XPU_ComplexHalf(a, b, c);
    return c;
};


namespace xpu {
    void kernel_add_XPU_ComplexFloat(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_add_XPU_ComplexFloat(a, b, c);
    return c;
};


namespace xpu {
    void kernel_add_XPU_ComplexDouble(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_add_XPU_ComplexDouble(a, b, c);
    return c;
};


namespace xpu {
    void kernel_add_XPU_Bool(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_add_XPU_Bool(a, b, c);
    return c;
};










namespace xpu {
    void kernel_sub_XPU_Int32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_sub_XPU_Int32(a, b, c);
    return c;
};


namespace xpu {
    void kernel_sub_XPU_Int64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_Int64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_sub_XPU_Int64(a, b, c);
    return c;
};


namespace xpu {
    void kernel_sub_XPU_Float32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_sub_XPU_Float32(a, b, c);
    return c;
};


namespace xpu {
    void kernel_sub_XPU_Float64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_sub_XPU_Float64(a, b, c);
    return c;
};


namespace xpu {
    void kernel_sub_XPU_Half(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_sub_XPU_Half(a, b, c);
    return c;
};


namespace xpu {
    void kernel_sub_XPU_Byte(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_sub_XPU_Byte(a, b, c);
    return c;
};


namespace xpu {
    void kernel_sub_XPU_Char(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_sub_XPU_Char(a, b, c);
    return c;
};


namespace xpu {
    void kernel_sub_XPU_Short(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_sub_XPU_Short(a, b, c);
    return c;
};


namespace xpu {
    void kernel_sub_XPU_Long(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_sub_XPU_Long(a, b, c);
    return c;
};


namespace xpu {
    void kernel_sub_XPU_ComplexHalf(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_sub_XPU_ComplexHalf(a, b, c);
    return c;
};


namespace xpu {
    void kernel_sub_XPU_ComplexFloat(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_sub_XPU_ComplexFloat(a, b, c);
    return c;
};


namespace xpu {
    void kernel_sub_XPU_ComplexDouble(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_sub_XPU_ComplexDouble(a, b, c);
    return c;
};


namespace xpu {
    void kernel_sub_XPU_Bool(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_sub_XPU_Bool(a, b, c);
    return c;
};










namespace xpu {
    void kernel_mul_XPU_Int32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_mul_XPU_Int32(a, b, c);
    return c;
};


namespace xpu {
    void kernel_mul_XPU_Int64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_Int64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_mul_XPU_Int64(a, b, c);
    return c;
};


namespace xpu {
    void kernel_mul_XPU_Float32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_mul_XPU_Float32(a, b, c);
    return c;
};


namespace xpu {
    void kernel_mul_XPU_Float64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_mul_XPU_Float64(a, b, c);
    return c;
};


namespace xpu {
    void kernel_mul_XPU_Half(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_mul_XPU_Half(a, b, c);
    return c;
};


namespace xpu {
    void kernel_mul_XPU_Byte(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_mul_XPU_Byte(a, b, c);
    return c;
};


namespace xpu {
    void kernel_mul_XPU_Char(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_mul_XPU_Char(a, b, c);
    return c;
};


namespace xpu {
    void kernel_mul_XPU_Short(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_mul_XPU_Short(a, b, c);
    return c;
};


namespace xpu {
    void kernel_mul_XPU_Long(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_mul_XPU_Long(a, b, c);
    return c;
};


namespace xpu {
    void kernel_mul_XPU_ComplexHalf(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_mul_XPU_ComplexHalf(a, b, c);
    return c;
};


namespace xpu {
    void kernel_mul_XPU_ComplexFloat(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_mul_XPU_ComplexFloat(a, b, c);
    return c;
};


namespace xpu {
    void kernel_mul_XPU_ComplexDouble(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_mul_XPU_ComplexDouble(a, b, c);
    return c;
};


namespace xpu {
    void kernel_mul_XPU_Bool(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_mul_XPU_Bool(a, b, c);
    return c;
};










namespace xpu {
    void kernel_matmul_XPU_Int32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_matmul_XPU_Int32(a, b, c);
    return c;
};


namespace xpu {
    void kernel_matmul_XPU_Int64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_Int64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_matmul_XPU_Int64(a, b, c);
    return c;
};


namespace xpu {
    void kernel_matmul_XPU_Float32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_matmul_XPU_Float32(a, b, c);
    return c;
};


namespace xpu {
    void kernel_matmul_XPU_Float64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_matmul_XPU_Float64(a, b, c);
    return c;
};


namespace xpu {
    void kernel_matmul_XPU_Half(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_matmul_XPU_Half(a, b, c);
    return c;
};


namespace xpu {
    void kernel_matmul_XPU_Byte(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_matmul_XPU_Byte(a, b, c);
    return c;
};


namespace xpu {
    void kernel_matmul_XPU_Char(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_matmul_XPU_Char(a, b, c);
    return c;
};


namespace xpu {
    void kernel_matmul_XPU_Short(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_matmul_XPU_Short(a, b, c);
    return c;
};


namespace xpu {
    void kernel_matmul_XPU_Long(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_matmul_XPU_Long(a, b, c);
    return c;
};


namespace xpu {
    void kernel_matmul_XPU_ComplexHalf(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_matmul_XPU_ComplexHalf(a, b, c);
    return c;
};


namespace xpu {
    void kernel_matmul_XPU_ComplexFloat(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_matmul_XPU_ComplexFloat(a, b, c);
    return c;
};


namespace xpu {
    void kernel_matmul_XPU_ComplexDouble(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_matmul_XPU_ComplexDouble(a, b, c);
    return c;
};


namespace xpu {
    void kernel_matmul_XPU_Bool(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_matmul_XPU_Bool(a, b, c);
    return c;
};










namespace xpu {
    void kernel_div_XPU_Int32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_div_XPU_Int32(a, b, c);
    return c;
};


namespace xpu {
    void kernel_div_XPU_Int64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_Int64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_div_XPU_Int64(a, b, c);
    return c;
};


namespace xpu {
    void kernel_div_XPU_Float32(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_div_XPU_Float32(a, b, c);
    return c;
};


namespace xpu {
    void kernel_div_XPU_Float64(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_div_XPU_Float64(a, b, c);
    return c;
};


namespace xpu {
    void kernel_div_XPU_Half(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_div_XPU_Half(a, b, c);
    return c;
};


namespace xpu {
    void kernel_div_XPU_Byte(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_div_XPU_Byte(a, b, c);
    return c;
};


namespace xpu {
    void kernel_div_XPU_Char(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_div_XPU_Char(a, b, c);
    return c;
};


namespace xpu {
    void kernel_div_XPU_Short(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_div_XPU_Short(a, b, c);
    return c;
};


namespace xpu {
    void kernel_div_XPU_Long(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_div_XPU_Long(a, b, c);
    return c;
};


namespace xpu {
    void kernel_div_XPU_ComplexHalf(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_div_XPU_ComplexHalf(a, b, c);
    return c;
};


namespace xpu {
    void kernel_div_XPU_ComplexFloat(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_div_XPU_ComplexFloat(a, b, c);
    return c;
};


namespace xpu {
    void kernel_div_XPU_ComplexDouble(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_div_XPU_ComplexDouble(a, b, c);
    return c;
};


namespace xpu {
    void kernel_div_XPU_Bool(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    xpu::kernel_div_XPU_Bool(a, b, c);
    return c;
};




