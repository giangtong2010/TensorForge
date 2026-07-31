#pragma once
#include "tensor.hpp"
#include "Dtype.hpp"






namespace cpu {
    template <typename T>
    void kernel_cpymem_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_CPU_Int32(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    cpu::kernel_cpymem_CPU<type>(dst, src, dtype, size);
};



namespace cpu {
    template <typename T>
    void kernel_cpymem_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_CPU_Float32(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    cpu::kernel_cpymem_CPU<type>(dst, src, dtype, size);
};



namespace cpu {
    template <typename T>
    void kernel_cpymem_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_CPU_Float64(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    cpu::kernel_cpymem_CPU<type>(dst, src, dtype, size);
};



namespace cpu {
    template <typename T>
    void kernel_cpymem_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_CPU_Byte(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    cpu::kernel_cpymem_CPU<type>(dst, src, dtype, size);
};



namespace cpu {
    template <typename T>
    void kernel_cpymem_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_CPU_Char(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    cpu::kernel_cpymem_CPU<type>(dst, src, dtype, size);
};



namespace cpu {
    template <typename T>
    void kernel_cpymem_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_CPU_Short(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    cpu::kernel_cpymem_CPU<type>(dst, src, dtype, size);
};



namespace cpu {
    template <typename T>
    void kernel_cpymem_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_CPU_Long(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    cpu::kernel_cpymem_CPU<type>(dst, src, dtype, size);
};



namespace cpu {
    template <typename T>
    void kernel_cpymem_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_CPU_ComplexFloat(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    cpu::kernel_cpymem_CPU<type>(dst, src, dtype, size);
};



namespace cpu {
    template <typename T>
    void kernel_cpymem_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_CPU_ComplexDouble(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    cpu::kernel_cpymem_CPU<type>(dst, src, dtype, size);
};



namespace cpu {
    template <typename T>
    void kernel_cpymem_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void cpymem_CPU_Bool(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    cpu::kernel_cpymem_CPU<type>(dst, src, dtype, size);
};











namespace cpu {
    template <typename T>
    void kernel_copy_from_host_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_CPU_Int32(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    cpu::kernel_copy_from_host_CPU<type>(dst, src, dtype, size);
};



namespace cpu {
    template <typename T>
    void kernel_copy_from_host_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_CPU_Float32(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    cpu::kernel_copy_from_host_CPU<type>(dst, src, dtype, size);
};



namespace cpu {
    template <typename T>
    void kernel_copy_from_host_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_CPU_Float64(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    cpu::kernel_copy_from_host_CPU<type>(dst, src, dtype, size);
};



namespace cpu {
    template <typename T>
    void kernel_copy_from_host_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_CPU_Byte(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    cpu::kernel_copy_from_host_CPU<type>(dst, src, dtype, size);
};



namespace cpu {
    template <typename T>
    void kernel_copy_from_host_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_CPU_Char(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    cpu::kernel_copy_from_host_CPU<type>(dst, src, dtype, size);
};



namespace cpu {
    template <typename T>
    void kernel_copy_from_host_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_CPU_Short(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    cpu::kernel_copy_from_host_CPU<type>(dst, src, dtype, size);
};



namespace cpu {
    template <typename T>
    void kernel_copy_from_host_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_CPU_Long(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    cpu::kernel_copy_from_host_CPU<type>(dst, src, dtype, size);
};



namespace cpu {
    template <typename T>
    void kernel_copy_from_host_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_CPU_ComplexFloat(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    cpu::kernel_copy_from_host_CPU<type>(dst, src, dtype, size);
};



namespace cpu {
    template <typename T>
    void kernel_copy_from_host_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_CPU_ComplexDouble(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    cpu::kernel_copy_from_host_CPU<type>(dst, src, dtype, size);
};



namespace cpu {
    template <typename T>
    void kernel_copy_from_host_CPU(void*, const void*, cpp20::Dtype, const size_t);
}

inline void copy_from_host_CPU_Bool(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    cpu::kernel_copy_from_host_CPU<type>(dst, src, dtype, size);
};






