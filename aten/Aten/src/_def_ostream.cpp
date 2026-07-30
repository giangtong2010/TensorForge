#include <iostream>
#include "Dtype.hpp"
#include "Device.hpp"
#include "tensor.hpp"

template <typename T>
void print_data(const uint8_t* data, size_t dim, const std::vector<int64_t>& shape, size_t& offset) {
    const T* ptr = reinterpret_cast<const T*>(data);

    std::cout <<"[";
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if (i) std::cout <<", " <<std::endl;
            std::cout <<ptr[offset++];
        }
    }
    else {
        for (size_t i = 0; i < shape[dim]; i++) {
            if(i) std::cout <<", " <<std::endl;
            print_data<T>(data, dim + 1, shape, offset);
        }
    }
    std::cout<<"]";
}

std::ostream& operator<<(std::ostream& os, const at::Tensor& tensor) {
    auto dtype = tensor.get_dtype();
    auto device = tensor.get_device();

    auto* data = tensor.data();
    auto shape = tensor.get_size();
    auto storage_offset = tensor.get_storage_offset();

    std::cout <<"tensor(";
    switch(dtype) {
        
        case cpp20::Dtype::Int32: {
            using type = cpp20::DtypeToCPPType<cpp20::Dtype::Int32>::type;
            print_data<type>(data, 0, shape, storage_offset);
            break;
        }
        
        case cpp20::Dtype::Float32: {
            using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float32>::type;
            print_data<type>(data, 0, shape, storage_offset);
            break;
        }
        
        case cpp20::Dtype::Float64: {
            using type = cpp20::DtypeToCPPType<cpp20::Dtype::Float64>::type;
            print_data<type>(data, 0, shape, storage_offset);
            break;
        }
        
        case cpp20::Dtype::Byte: {
            using type = cpp20::DtypeToCPPType<cpp20::Dtype::Byte>::type;
            print_data<type>(data, 0, shape, storage_offset);
            break;
        }
        
        case cpp20::Dtype::Char: {
            using type = cpp20::DtypeToCPPType<cpp20::Dtype::Char>::type;
            print_data<type>(data, 0, shape, storage_offset);
            break;
        }
        
        case cpp20::Dtype::Short: {
            using type = cpp20::DtypeToCPPType<cpp20::Dtype::Short>::type;
            print_data<type>(data, 0, shape, storage_offset);
            break;
        }
        
        case cpp20::Dtype::Long: {
            using type = cpp20::DtypeToCPPType<cpp20::Dtype::Long>::type;
            print_data<type>(data, 0, shape, storage_offset);
            break;
        }
        
        case cpp20::Dtype::ComplexFloat: {
            using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexFloat>::type;
            print_data<type>(data, 0, shape, storage_offset);
            break;
        }
        
        case cpp20::Dtype::ComplexDouble: {
            using type = cpp20::DtypeToCPPType<cpp20::Dtype::ComplexDouble>::type;
            print_data<type>(data, 0, shape, storage_offset);
            break;
        }
        
        case cpp20::Dtype::Bool: {
            using type = cpp20::DtypeToCPPType<cpp20::Dtype::Bool>::type;
            print_data<type>(data, 0, shape, storage_offset);
            break;
        }
        
    }
    std::cout <<"), dtype=" <<dtype <<" device=" <<device <<std::endl; 
    return os;
}