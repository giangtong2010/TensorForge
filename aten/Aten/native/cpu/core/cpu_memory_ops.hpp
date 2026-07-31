#pragma once
#include "tensor.hpp"
#include "Dtype.hpp"






namespace cpu {
    template <typename T>
    void kernel_contiguous_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_CPU_Int32(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    cpu::kernel_contiguous_CPU<type>(dst, src);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_contiguous_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_CPU_Float32(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    cpu::kernel_contiguous_CPU<type>(dst, src);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_contiguous_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_CPU_Float64(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    cpu::kernel_contiguous_CPU<type>(dst, src);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_contiguous_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_CPU_Byte(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    cpu::kernel_contiguous_CPU<type>(dst, src);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_contiguous_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_CPU_Char(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    cpu::kernel_contiguous_CPU<type>(dst, src);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_contiguous_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_CPU_Short(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    cpu::kernel_contiguous_CPU<type>(dst, src);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_contiguous_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_CPU_Long(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    cpu::kernel_contiguous_CPU<type>(dst, src);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_contiguous_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_CPU_ComplexFloat(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    cpu::kernel_contiguous_CPU<type>(dst, src);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_contiguous_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_CPU_ComplexDouble(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    cpu::kernel_contiguous_CPU<type>(dst, src);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_contiguous_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_CPU_Bool(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    cpu::kernel_contiguous_CPU<type>(dst, src);
    return c;
};











namespace cpu {
    template <typename T>
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_CPU_Int32(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_CPU_Float32(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_CPU_Float64(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_CPU_Byte(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_CPU_Char(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_CPU_Short(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_CPU_Long(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_CPU_ComplexFloat(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_CPU_ComplexDouble(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_CPU_Bool(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
    return c;
};






