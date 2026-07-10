#pragma once
#include "tensor.hpp"
#include "Dtype.hpp"






namespace xpu {
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Half>::type;
    xpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexHalf>::type;
    xpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor contiguous_CPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;
};











namespace xpu {
    template <typename T>
    void kernel_add_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_add_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_add_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_add_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Half>::type;
    xpu::kernel_add_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_add_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_add_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_add_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_add_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexHalf>::type;
    xpu::kernel_add_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_add_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_add_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor add_CPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_add_CPU<type>(a, b, c);
    return c;
};











namespace xpu {
    template <typename T>
    void kernel_sub_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_sub_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_sub_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_sub_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Half>::type;
    xpu::kernel_sub_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_sub_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_sub_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_sub_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_sub_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexHalf>::type;
    xpu::kernel_sub_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_sub_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_sub_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor sub_CPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_sub_CPU<type>(a, b, c);
    return c;
};











namespace xpu {
    template <typename T>
    void kernel_mul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_mul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_mul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_mul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Half>::type;
    xpu::kernel_mul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_mul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_mul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_mul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_mul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexHalf>::type;
    xpu::kernel_mul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_mul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_mul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor mul_CPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_mul_CPU<type>(a, b, c);
    return c;
};











namespace xpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Half>::type;
    xpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexHalf>::type;
    xpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor matmul_CPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};











namespace xpu {
    template <typename T>
    void kernel_div_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_div_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_div_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_div_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_Half(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Half>::type;
    xpu::kernel_div_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_div_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_div_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_div_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_div_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_ComplexHalf(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexHalf>::type;
    xpu::kernel_div_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_div_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_div_CPU<type>(a, b, c);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}
inline at::Tensor div_CPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_div_CPU<type>(a, b, c);
    return c;
};






