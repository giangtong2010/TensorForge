#pragma once
#include "tensor.hpp"
#include "Dtype.hpp"








namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_XPU_Int32(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_contiguous_XPU<type>(dst, src);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_XPU_Float32(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_contiguous_XPU<type>(dst, src);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_XPU_Float64(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_contiguous_XPU<type>(dst, src);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_XPU_Byte(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_contiguous_XPU<type>(dst, src);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_XPU_Char(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_contiguous_XPU<type>(dst, src);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_XPU_Short(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_contiguous_XPU<type>(dst, src);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_XPU_Long(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_contiguous_XPU<type>(dst, src);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_XPU_ComplexFloat(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_contiguous_XPU<type>(dst, src);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_XPU_ComplexDouble(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_contiguous_XPU<type>(dst, src);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_contiguous_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor contiguous_XPU_Bool(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_contiguous_XPU<type>(dst, src);
    return c;
};











namespace xpu {
    template <typename T>
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_XPU_Int32(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_XPU_Float32(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_XPU_Float64(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_XPU_Byte(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_XPU_Char(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_XPU_Short(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_XPU_Long(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_XPU_ComplexFloat(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_XPU_ComplexDouble(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
}

inline at::Tensor copy_XPU_Bool(at::Tensor& dst, const at::Tensor& src) {
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
    return c;
};




