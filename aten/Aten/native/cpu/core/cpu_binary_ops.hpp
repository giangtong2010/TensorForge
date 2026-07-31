#pragma once
#include "tensor.hpp"
#include "Dtype.hpp"
#include "_tensor_iterator.hpp"






namespace cpu {
    template <typename T>
    void kernel_add_CPU(const at::TensorIterator& iter);
}

inline at::Tensor add_CPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    cpu::kernel_add_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_add_CPU(const at::TensorIterator& iter);
}

inline at::Tensor add_CPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    cpu::kernel_add_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_add_CPU(const at::TensorIterator& iter);
}

inline at::Tensor add_CPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    cpu::kernel_add_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_add_CPU(const at::TensorIterator& iter);
}

inline at::Tensor add_CPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    cpu::kernel_add_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_add_CPU(const at::TensorIterator& iter);
}

inline at::Tensor add_CPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    cpu::kernel_add_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_add_CPU(const at::TensorIterator& iter);
}

inline at::Tensor add_CPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    cpu::kernel_add_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_add_CPU(const at::TensorIterator& iter);
}

inline at::Tensor add_CPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    cpu::kernel_add_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_add_CPU(const at::TensorIterator& iter);
}

inline at::Tensor add_CPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    cpu::kernel_add_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_add_CPU(const at::TensorIterator& iter);
}

inline at::Tensor add_CPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    cpu::kernel_add_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_add_CPU(const at::TensorIterator& iter);
}

inline at::Tensor add_CPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    cpu::kernel_add_CPU<type>(iter);
    return c;
};











namespace cpu {
    template <typename T>
    void kernel_sub_CPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_CPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    cpu::kernel_sub_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_sub_CPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_CPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    cpu::kernel_sub_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_sub_CPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_CPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    cpu::kernel_sub_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_sub_CPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_CPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    cpu::kernel_sub_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_sub_CPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_CPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    cpu::kernel_sub_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_sub_CPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_CPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    cpu::kernel_sub_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_sub_CPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_CPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    cpu::kernel_sub_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_sub_CPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_CPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    cpu::kernel_sub_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_sub_CPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_CPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    cpu::kernel_sub_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_sub_CPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_CPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    cpu::kernel_sub_CPU<type>(iter);
    return c;
};











namespace cpu {
    template <typename T>
    void kernel_mul_CPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_CPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    cpu::kernel_mul_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_mul_CPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_CPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    cpu::kernel_mul_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_mul_CPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_CPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    cpu::kernel_mul_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_mul_CPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_CPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    cpu::kernel_mul_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_mul_CPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_CPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    cpu::kernel_mul_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_mul_CPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_CPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    cpu::kernel_mul_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_mul_CPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_CPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    cpu::kernel_mul_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_mul_CPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_CPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    cpu::kernel_mul_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_mul_CPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_CPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    cpu::kernel_mul_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_mul_CPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_CPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    cpu::kernel_mul_CPU<type>(iter);
    return c;
};











namespace cpu {
    template <typename T>
    void kernel_div_CPU(const at::TensorIterator& iter);
}

inline at::Tensor div_CPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    cpu::kernel_div_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_div_CPU(const at::TensorIterator& iter);
}

inline at::Tensor div_CPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    cpu::kernel_div_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_div_CPU(const at::TensorIterator& iter);
}

inline at::Tensor div_CPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    cpu::kernel_div_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_div_CPU(const at::TensorIterator& iter);
}

inline at::Tensor div_CPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    cpu::kernel_div_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_div_CPU(const at::TensorIterator& iter);
}

inline at::Tensor div_CPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    cpu::kernel_div_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_div_CPU(const at::TensorIterator& iter);
}

inline at::Tensor div_CPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    cpu::kernel_div_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_div_CPU(const at::TensorIterator& iter);
}

inline at::Tensor div_CPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    cpu::kernel_div_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_div_CPU(const at::TensorIterator& iter);
}

inline at::Tensor div_CPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    cpu::kernel_div_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_div_CPU(const at::TensorIterator& iter);
}

inline at::Tensor div_CPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    cpu::kernel_div_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_div_CPU(const at::TensorIterator& iter);
}

inline at::Tensor div_CPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    cpu::kernel_div_CPU<type>(iter);
    return c;
};











namespace cpu {
    template <typename T>
    void kernel_pow_CPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_CPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    cpu::kernel_pow_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_pow_CPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_CPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    cpu::kernel_pow_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_pow_CPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_CPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    cpu::kernel_pow_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_pow_CPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_CPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    cpu::kernel_pow_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_pow_CPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_CPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    cpu::kernel_pow_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_pow_CPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_CPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    cpu::kernel_pow_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_pow_CPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_CPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    cpu::kernel_pow_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_pow_CPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_CPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    cpu::kernel_pow_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_pow_CPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_CPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    cpu::kernel_pow_CPU<type>(iter);
    return c;
};



namespace cpu {
    template <typename T>
    void kernel_pow_CPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_CPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    cpu::kernel_pow_CPU<type>(iter);
    return c;
};






