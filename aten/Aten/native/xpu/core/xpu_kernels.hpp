#pragma once
#include "tensor.hpp"
#include "Dtype.hpp"
#include "_tensor_iterator.hpp"








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
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_XPU_Int32(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
}




namespace xpu {
    
    template <typename T>
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_XPU_Float32(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
}




namespace xpu {
    
    template <typename T>
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_XPU_Float64(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
}




namespace xpu {
    
    template <typename T>
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_XPU_Byte(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
}




namespace xpu {
    
    template <typename T>
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_XPU_Char(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
}




namespace xpu {
    
    template <typename T>
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_XPU_Short(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
}




namespace xpu {
    
    template <typename T>
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_XPU_Long(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
}




namespace xpu {
    
    template <typename T>
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_XPU_ComplexFloat(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
}




namespace xpu {
    
    template <typename T>
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_XPU_ComplexDouble(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
}




namespace xpu {
    
    template <typename T>
    void kernel_copy_XPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_XPU_Bool(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_copy_XPU<type>(dst, src);
}












namespace xpu {
    
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void cpymem_XPU_Int32(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_cpymem_XPU<type>(dst, src, dtype, size);
}




namespace xpu {
    
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void cpymem_XPU_Float32(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_cpymem_XPU<type>(dst, src, dtype, size);
}




namespace xpu {
    
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void cpymem_XPU_Float64(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_cpymem_XPU<type>(dst, src, dtype, size);
}




namespace xpu {
    
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void cpymem_XPU_Byte(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_cpymem_XPU<type>(dst, src, dtype, size);
}




namespace xpu {
    
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void cpymem_XPU_Char(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_cpymem_XPU<type>(dst, src, dtype, size);
}




namespace xpu {
    
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void cpymem_XPU_Short(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_cpymem_XPU<type>(dst, src, dtype, size);
}




namespace xpu {
    
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void cpymem_XPU_Long(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_cpymem_XPU<type>(dst, src, dtype, size);
}




namespace xpu {
    
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void cpymem_XPU_ComplexFloat(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_cpymem_XPU<type>(dst, src, dtype, size);
}




namespace xpu {
    
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void cpymem_XPU_ComplexDouble(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_cpymem_XPU<type>(dst, src, dtype, size);
}




namespace xpu {
    
    template <typename T>
    void kernel_cpymem_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void cpymem_XPU_Bool(void* dst, const void* src, cpp20::Dtype dtype, const size_t size) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_cpymem_XPU<type>(dst, src, dtype, size);
}












namespace xpu {
    
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void copy_from_host_XPU_Int32(void* dst, const void* src, cpp20::Dtype dtype, const size_t numel) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    xpu::kernel_copy_from_host_XPU<type>(dst, src, dtype, numel);
}




namespace xpu {
    
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void copy_from_host_XPU_Float32(void* dst, const void* src, cpp20::Dtype dtype, const size_t numel) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    xpu::kernel_copy_from_host_XPU<type>(dst, src, dtype, numel);
}




namespace xpu {
    
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void copy_from_host_XPU_Float64(void* dst, const void* src, cpp20::Dtype dtype, const size_t numel) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    xpu::kernel_copy_from_host_XPU<type>(dst, src, dtype, numel);
}




namespace xpu {
    
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void copy_from_host_XPU_Byte(void* dst, const void* src, cpp20::Dtype dtype, const size_t numel) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    xpu::kernel_copy_from_host_XPU<type>(dst, src, dtype, numel);
}




namespace xpu {
    
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void copy_from_host_XPU_Char(void* dst, const void* src, cpp20::Dtype dtype, const size_t numel) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    xpu::kernel_copy_from_host_XPU<type>(dst, src, dtype, numel);
}




namespace xpu {
    
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void copy_from_host_XPU_Short(void* dst, const void* src, cpp20::Dtype dtype, const size_t numel) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    xpu::kernel_copy_from_host_XPU<type>(dst, src, dtype, numel);
}




namespace xpu {
    
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void copy_from_host_XPU_Long(void* dst, const void* src, cpp20::Dtype dtype, const size_t numel) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    xpu::kernel_copy_from_host_XPU<type>(dst, src, dtype, numel);
}




namespace xpu {
    
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void copy_from_host_XPU_ComplexFloat(void* dst, const void* src, cpp20::Dtype dtype, const size_t numel) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    xpu::kernel_copy_from_host_XPU<type>(dst, src, dtype, numel);
}




namespace xpu {
    
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void copy_from_host_XPU_ComplexDouble(void* dst, const void* src, cpp20::Dtype dtype, const size_t numel) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    xpu::kernel_copy_from_host_XPU<type>(dst, src, dtype, numel);
}




namespace xpu {
    
    template <typename T>
    void kernel_copy_from_host_XPU(void*, const void*, cpp20::Dtype, const size_t);
    
}


inline void copy_from_host_XPU_Bool(void* dst, const void* src, cpp20::Dtype dtype, const size_t numel) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    xpu::kernel_copy_from_host_XPU<type>(dst, src, dtype, numel);
}





