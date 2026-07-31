#include <iostream>
#include "Dtype.hpp"
#include "Device.hpp"
#include "_tensor.hpp"

template <typename T>
void print_data(
    std::ostream& os, const uint8_t* data, size_t dim, size_t offset, 
    const std::vector<int64_t>& shape, const std::vector<int64_t>& stride
) {
    const T* ptr = reinterpret_cast<const T*>(data);

    os <<"[";
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if (i) os <<", ";
            os << ptr[offset + i * stride[dim]];        
        }
    }
    else {
        for (size_t i = 0; i < shape[dim]; i++) {
            if(i) os <<", " <<std::endl;
            auto child_offset = offset + i * stride[dim];
            print_data<T>(os, data, dim + 1, child_offset, shape, stride);
        }
    }
    os<< "]";
}



#define PRINT_CASE(DTYPE)                                                                   \
case cpp20::Dtype::DTYPE: {                                                                 \
    using type = cpp20::DtypeToCPPType<cpp20::Dtype::DTYPE>::type;                          \
    print_data<type>(os, data, 0, storage_offset, shape, stride);                           \
    break;                                                                                  \
}

std::ostream& operator<<(std::ostream& os, const at::Tensor& tensor) {
    auto dtype = tensor.get_dtype();
    auto device = tensor.get_device();

    auto* data = tensor.data();
    auto shape = tensor.get_size();
    auto stride = tensor.get_stride();
    auto storage_offset = tensor.get_storage_offset();

    os <<"tensor(";
    switch(dtype) {
        #define X(T) PRINT_CASE(T)
        #include "dtype.def"
        #undef X        
    }

    os <<"), dtype=" << dtype
       << ", device=" << device;

    return os;
}

std::ostream& operator<<(std::ostream& os, at::Tensor& tensor) {
    auto dtype = tensor.get_dtype();
    auto device = tensor.get_device();

    auto* data = tensor.data();
    auto shape = tensor.get_size();
    auto stride = tensor.get_stride();
    auto storage_offset = tensor.get_storage_offset();

    os <<"tensor(";
    switch (dtype) {
        #define X(T) PRINT_CASE(T)
        #include "dtype.def"
        #undef X
    }

    os <<"), dtype=" <<dtype
        <<", device=" <<device;
    return os;
}

#undef PRIN_CASE
