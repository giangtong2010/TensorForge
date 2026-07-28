#pragma once
#include "tensor.hpp"
#include "Dtype.hpp"
#include "_tensor_iterator.hpp"






namespace cpu {
    
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
    
}


inline at::Tensor contiguous_CPU_Int32(const at::Tensor& a, const at::Tensor& b) {
    
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    cpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;

    
};




namespace cpu {
    
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
    
}


inline at::Tensor contiguous_CPU_Float32(const at::Tensor& a, const at::Tensor& b) {
    
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    cpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;

    
};




namespace cpu {
    
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
    
}


inline at::Tensor contiguous_CPU_Float64(const at::Tensor& a, const at::Tensor& b) {
    
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    cpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;

    
};




namespace cpu {
    
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
    
}


inline at::Tensor contiguous_CPU_Byte(const at::Tensor& a, const at::Tensor& b) {
    
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    cpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;

    
};




namespace cpu {
    
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
    
}


inline at::Tensor contiguous_CPU_Char(const at::Tensor& a, const at::Tensor& b) {
    
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    cpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;

    
};




namespace cpu {
    
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
    
}


inline at::Tensor contiguous_CPU_Short(const at::Tensor& a, const at::Tensor& b) {
    
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    cpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;

    
};




namespace cpu {
    
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
    
}


inline at::Tensor contiguous_CPU_Long(const at::Tensor& a, const at::Tensor& b) {
    
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    cpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;

    
};




namespace cpu {
    
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
    
}


inline at::Tensor contiguous_CPU_ComplexFloat(const at::Tensor& a, const at::Tensor& b) {
    
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    cpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;

    
};




namespace cpu {
    
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
    
}


inline at::Tensor contiguous_CPU_ComplexDouble(const at::Tensor& a, const at::Tensor& b) {
    
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    cpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;

    
};




namespace cpu {
    
    template <typename T>
    void kernel_contiguous_CPU(const at::Tensor&, const at::Tensor&, at::Tensor&);
    
}


inline at::Tensor contiguous_CPU_Bool(const at::Tensor& a, const at::Tensor& b) {
    
    at::Tensor c;
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    cpu::kernel_contiguous_CPU<type>(a, b, c);
    return c;

    
};












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
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_CPU_Int32(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
}




namespace cpu {
    
    template <typename T>
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_CPU_Float32(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
}




namespace cpu {
    
    template <typename T>
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_CPU_Float64(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
}




namespace cpu {
    
    template <typename T>
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_CPU_Byte(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
}




namespace cpu {
    
    template <typename T>
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_CPU_Char(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
}




namespace cpu {
    
    template <typename T>
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_CPU_Short(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
}




namespace cpu {
    
    template <typename T>
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_CPU_Long(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
}




namespace cpu {
    
    template <typename T>
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_CPU_ComplexFloat(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
}




namespace cpu {
    
    template <typename T>
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_CPU_ComplexDouble(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
}




namespace cpu {
    
    template <typename T>
    void kernel_copy_CPU(at::Tensor&, const at::Tensor&);
    
}


inline void copy_CPU_Bool(at::Tensor& dst, const at::Tensor& src) {
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
    cpu::kernel_copy_CPU<type>(dst, src);
}







