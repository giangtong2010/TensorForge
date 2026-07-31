#pragma once
#include "tensor.hpp"
#include "Dtype.hpp"






namespace cpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}

inline at::Tensor matmul_CPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    cpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}

inline at::Tensor matmul_CPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    cpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}

inline at::Tensor matmul_CPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    cpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}

inline at::Tensor matmul_CPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    cpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}

inline at::Tensor matmul_CPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    cpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}

inline at::Tensor matmul_CPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    cpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}

inline at::Tensor matmul_CPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    cpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}

inline at::Tensor matmul_CPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    cpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}

inline at::Tensor matmul_CPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    cpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_matmul_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
}

inline at::Tensor matmul_CPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    cpu::kernel_matmul_CPU<type>(a, b, c);
    return c;
};






