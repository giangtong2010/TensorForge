#pragma once
#include "tensor.hpp"
#include "Dtype.hpp"








namespace xpu {
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_XPU_Int32(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu
::kernel_cpymem_XPU<type>(dst, src, dtype, size);
};



namespace xpu {
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_XPU_Float32(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu
::kernel_cpymem_XPU<type>(dst, src, dtype, size);
};



namespace xpu {
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_XPU_Float64(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu
::kernel_cpymem_XPU<type>(dst, src, dtype, size);
};



namespace xpu {
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_XPU_Byte(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu
::kernel_cpymem_XPU<type>(dst, src, dtype, size);
};



namespace xpu {
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_XPU_Char(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu
::kernel_cpymem_XPU<type>(dst, src, dtype, size);
};



namespace xpu {
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_XPU_Short(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu
::kernel_cpymem_XPU<type>(dst, src, dtype, size);
};



namespace xpu {
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_XPU_Long(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu
::kernel_cpymem_XPU<type>(dst, src, dtype, size);
};



namespace xpu {
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_XPU_ComplexFloat(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu
::kernel_cpymem_XPU<type>(dst, src, dtype, size);
};



namespace xpu {
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_XPU_ComplexDouble(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu
::kernel_cpymem_XPU<type>(dst, src, dtype, size);
};



namespace xpu {
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_XPU_Bool(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu
::kernel_cpymem_XPU<type>(dst, src, dtype, size);
};











namespace xpu {
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_XPU_Int32(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu
::kernel_copy_from_host_XPU<type>(dst, src, dtype, size);
};



namespace xpu {
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_XPU_Float32(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu
::kernel_copy_from_host_XPU<type>(dst, src, dtype, size);
};



namespace xpu {
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_XPU_Float64(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu
::kernel_copy_from_host_XPU<type>(dst, src, dtype, size);
};



namespace xpu {
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_XPU_Byte(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu
::kernel_copy_from_host_XPU<type>(dst, src, dtype, size);
};



namespace xpu {
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_XPU_Char(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu
::kernel_copy_from_host_XPU<type>(dst, src, dtype, size);
};



namespace xpu {
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_XPU_Short(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu
::kernel_copy_from_host_XPU<type>(dst, src, dtype, size);
};



namespace xpu {
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_XPU_Long(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu
::kernel_copy_from_host_XPU<type>(dst, src, dtype, size);
};



namespace xpu {
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_XPU_ComplexFloat(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu
::kernel_copy_from_host_XPU<type>(dst, src, dtype, size);
};



namespace xpu {
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_XPU_ComplexDouble(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu
::kernel_copy_from_host_XPU<type>(dst, src, dtype, size);
};



namespace xpu {
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_XPU_Bool(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu
::kernel_copy_from_host_XPU<type>(dst, src, dtype, size);
};




