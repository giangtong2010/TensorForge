#pragma once
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <complex>

namespace cpp20 {
    struct Half {
        uint16_t bits;
    };

    enum class Dtype {
        
        Int32,
        
        Float32,
        
        Float64,
        
        Half,
        
        Byte,
        
        Char,
        
        Short,
        
        Long,
        
        ComplexHalf,
        
        ComplexFloat,
        
        ComplexDouble,
        
        Bool,
        
        count=12,
    };

    inline size_t dtype_size(Dtype dtype) {
        switch (dtype) {
            
            case Dtype::Int32: return sizeof(int32_t);
            
            case Dtype::Float32: return sizeof(float);
            
            case Dtype::Float64: return sizeof(double);
            
            case Dtype::Half: return sizeof(Half);
            
            case Dtype::Byte: return sizeof(uint8_t);
            
            case Dtype::Char: return sizeof(int8_t);
            
            case Dtype::Short: return sizeof(int16_t);
            
            case Dtype::Long: return sizeof(int64_t);
            
            case Dtype::ComplexHalf: return sizeof(std::complex<Half>);
            
            case Dtype::ComplexFloat: return sizeof(std::complex<float>);
            
            case Dtype::ComplexDouble: return sizeof(std::complex<double>);
            
            case Dtype::Bool: return sizeof(bool);
            
            
            default:
                throw std::invalid_argument("Unsupported dtype");
        }
    }

    template<cpp20::Dtype type>
    struct DtypeToCPPType;

    
    template <>
    struct DtypeToCPPType<cpp20::Dtype::Int32> {
        using type = int32_t;
    };
    
    template <>
    struct DtypeToCPPType<cpp20::Dtype::Float32> {
        using type = float;
    };
    
    template <>
    struct DtypeToCPPType<cpp20::Dtype::Float64> {
        using type = double;
    };
    
    template <>
    struct DtypeToCPPType<cpp20::Dtype::Half> {
        using type = Half;
    };
    
    template <>
    struct DtypeToCPPType<cpp20::Dtype::Byte> {
        using type = uint8_t;
    };
    
    template <>
    struct DtypeToCPPType<cpp20::Dtype::Char> {
        using type = int8_t;
    };
    
    template <>
    struct DtypeToCPPType<cpp20::Dtype::Short> {
        using type = int16_t;
    };
    
    template <>
    struct DtypeToCPPType<cpp20::Dtype::Long> {
        using type = int64_t;
    };
    
    template <>
    struct DtypeToCPPType<cpp20::Dtype::ComplexHalf> {
        using type = std::complex<Half>;
    };
    
    template <>
    struct DtypeToCPPType<cpp20::Dtype::ComplexFloat> {
        using type = std::complex<float>;
    };
    
    template <>
    struct DtypeToCPPType<cpp20::Dtype::ComplexDouble> {
        using type = std::complex<double>;
    };
    
    template <>
    struct DtypeToCPPType<cpp20::Dtype::Bool> {
        using type = bool;
    };
    

    template<typename T>
    struct CPPTypeToDtype;

    
    template <>
    struct CPPTypeToDtype<int32_t> {
        static constexpr Dtype value = Dtype::Int32;
    };
    
    template <>
    struct CPPTypeToDtype<float> {
        static constexpr Dtype value = Dtype::Float32;
    };
    
    template <>
    struct CPPTypeToDtype<double> {
        static constexpr Dtype value = Dtype::Float64;
    };
    
    template <>
    struct CPPTypeToDtype<Half> {
        static constexpr Dtype value = Dtype::Half;
    };
    
    template <>
    struct CPPTypeToDtype<uint8_t> {
        static constexpr Dtype value = Dtype::Byte;
    };
    
    template <>
    struct CPPTypeToDtype<int8_t> {
        static constexpr Dtype value = Dtype::Char;
    };
    
    template <>
    struct CPPTypeToDtype<int16_t> {
        static constexpr Dtype value = Dtype::Short;
    };
    
    template <>
    struct CPPTypeToDtype<int64_t> {
        static constexpr Dtype value = Dtype::Long;
    };
    
    template <>
    struct CPPTypeToDtype<std::complex<Half>> {
        static constexpr Dtype value = Dtype::ComplexHalf;
    };
    
    template <>
    struct CPPTypeToDtype<std::complex<float>> {
        static constexpr Dtype value = Dtype::ComplexFloat;
    };
    
    template <>
    struct CPPTypeToDtype<std::complex<double>> {
        static constexpr Dtype value = Dtype::ComplexDouble;
    };
    
    template <>
    struct CPPTypeToDtype<bool> {
        static constexpr Dtype value = Dtype::Bool;
    };
    
}