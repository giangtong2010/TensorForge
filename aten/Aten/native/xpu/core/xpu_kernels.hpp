#pragma once
#include "tensor.hpp"
#include "Dtype.hpp"








namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_contiguous_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_contiguous_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_contiguous_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Half>::type;
    xpu::kernel_contiguous_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_contiguous_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_contiguous_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_contiguous_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_contiguous_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexHalf>::type;
    xpu::kernel_contiguous_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_contiguous_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_contiguous_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_XPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_contiguous_XPU<type>(a, b, c);
    return c;
};











namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_add_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_add_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_add_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Half>::type;
    xpu::kernel_add_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_add_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_add_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_add_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_add_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexHalf>::type;
    xpu::kernel_add_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_add_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_add_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_XPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_add_XPU<type>(a, b, c);
    return c;
};











namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_sub_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_sub_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_sub_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Half>::type;
    xpu::kernel_sub_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_sub_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_sub_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_sub_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_sub_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexHalf>::type;
    xpu::kernel_sub_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_sub_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_sub_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_XPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_sub_XPU<type>(a, b, c);
    return c;
};











namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_mul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_mul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_mul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Half>::type;
    xpu::kernel_mul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_mul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_mul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_mul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_mul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexHalf>::type;
    xpu::kernel_mul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_mul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_mul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_XPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_mul_XPU<type>(a, b, c);
    return c;
};











namespace xpu {
    template <typename T>
    void kernel_matmul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_matmul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_matmul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_matmul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Half>::type;
    xpu::kernel_matmul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_matmul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_matmul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_matmul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_matmul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexHalf>::type;
    xpu::kernel_matmul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_matmul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_matmul_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_XPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_matmul_XPU<type>(a, b, c);
    return c;
};











namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_div_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_div_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_div_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Half>::type;
    xpu::kernel_div_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_div_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_div_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_div_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_div_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexHalf>::type;
    xpu::kernel_div_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_div_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_div_XPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_XPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_div_XPU<type>(a, b, c);
    return c;
};




