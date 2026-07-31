#pragma once
#include "tensor.hpp"
#include "Dtype.hpp"
#include "_tensor_iterator.hpp"








namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::TensorIterator& iter);
}

inline at::Tensor add_XPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_add_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::TensorIterator& iter);
}

inline at::Tensor add_XPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_add_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::TensorIterator& iter);
}

inline at::Tensor add_XPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_add_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::TensorIterator& iter);
}

inline at::Tensor add_XPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_add_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::TensorIterator& iter);
}

inline at::Tensor add_XPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_add_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::TensorIterator& iter);
}

inline at::Tensor add_XPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_add_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::TensorIterator& iter);
}

inline at::Tensor add_XPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_add_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::TensorIterator& iter);
}

inline at::Tensor add_XPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_add_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::TensorIterator& iter);
}

inline at::Tensor add_XPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_add_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_add_XPU(const at::TensorIterator& iter);
}

inline at::Tensor add_XPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_add_XPU<type>(iter);
    return c;
};











namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_XPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_sub_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_XPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_sub_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_XPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_sub_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_XPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_sub_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_XPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_sub_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_XPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_sub_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_XPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_sub_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_XPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_sub_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_XPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_sub_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_sub_XPU(const at::TensorIterator& iter);
}

inline at::Tensor sub_XPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_sub_XPU<type>(iter);
    return c;
};











namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_XPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_mul_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_XPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_mul_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_XPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_mul_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_XPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_mul_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_XPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_mul_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_XPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_mul_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_XPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_mul_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_XPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_mul_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_XPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_mul_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_mul_XPU(const at::TensorIterator& iter);
}

inline at::Tensor mul_XPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_mul_XPU<type>(iter);
    return c;
};











namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::TensorIterator& iter);
}

inline at::Tensor div_XPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_div_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::TensorIterator& iter);
}

inline at::Tensor div_XPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_div_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::TensorIterator& iter);
}

inline at::Tensor div_XPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_div_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::TensorIterator& iter);
}

inline at::Tensor div_XPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_div_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::TensorIterator& iter);
}

inline at::Tensor div_XPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_div_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::TensorIterator& iter);
}

inline at::Tensor div_XPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_div_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::TensorIterator& iter);
}

inline at::Tensor div_XPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_div_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::TensorIterator& iter);
}

inline at::Tensor div_XPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_div_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::TensorIterator& iter);
}

inline at::Tensor div_XPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_div_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_div_XPU(const at::TensorIterator& iter);
}

inline at::Tensor div_XPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_div_XPU<type>(iter);
    return c;
};











namespace xpu {
    template <typename T>
    void kernel_pow_XPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_XPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_pow_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_pow_XPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_XPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_pow_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_pow_XPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_XPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_pow_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_pow_XPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_XPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_pow_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_pow_XPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_XPU_Char(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_pow_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_pow_XPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_XPU_Short(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_pow_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_pow_XPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_XPU_Long(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_pow_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_pow_XPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_XPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_pow_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_pow_XPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_XPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_pow_XPU<type>(iter);
    return c;
};



namespace xpu {
    template <typename T>
    void kernel_pow_XPU(const at::TensorIterator& iter);
}

inline at::Tensor pow_XPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    at::TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.infer();

    at::Tensor c = at::Tensor::empty(iter.get_commo_shape(), iter.get_out_dtype(), a.get_device());
    iter.add_output(c);

    iter.build();

    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_pow_XPU<type>(iter);
    return c;
};




