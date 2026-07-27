#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#include <tuple>
#include "Dtype.hpp"
#include "utils.hpp"
#include "_tensor.hpp"

namespace at {
    enum class OperandTypes {
        Input, Output
    };

    class OffsetCalculator {
        std::vector<int64_t> _shape;
        std::vector<int64_t> _stride;
        size_t _storage_offset;

    public:
        OffsetCalculator(
            size_t storage_offset,
            std::vector<int64_t> shape,
            std::vector<int64_t> stride
        )
            : _shape(std::move(shape)),
            _stride(std::move(stride)),
            _storage_offset(storage_offset) {}

        size_t operator()(size_t linear_indx) {
            return cpp20::compute_offset(
                _storage_offset,
                linear_indx,
                _shape,
                _stride
            );
        }
    };

    struct OperandInfo {
        Tensor& _head_tensor;
        uint8_t* _head_pointer;
        std::vector<int64_t> _head_shape;
        std::vector<int64_t> _head_stride;
        cpp20::Dtype _head_dtype;
        cpp20::Device _head_device;

        size_t _storage_offset = 0;
        OperandTypes _type;

        ~OperandInfo() = default;
        OperandInfo(Tensor& a, OperandTypes type) :
            _head_tensor(a),
            _storage_offset(a.get_storage_offset()),
            _head_pointer(a.data()),
            _head_shape(a.get_size()),
            _head_stride(a.get_stride()),
            _head_dtype(a.get_dtype()),
            _head_device(a.get_device()),
            _type(type) {};
    };

    class TensorIterator {
        std::vector<OperandInfo> _operands;
        std::vector<int64_t> _commo_shape;
        cpp20::Dtype out_dtype;
        cpp20::Device out_device;
        int _num_in = 0;
        int _num_out = 0;

        void validate_operands();
        void coalesce_dimensions();
        void coalesce_dtype();
        void broad_cast_shape();
        void broad_cast_stride();
        void check_dev();

    public:
        TensorIterator() = default;
        ~TensorIterator() = default;

        void add_output(Tensor& _out_tensor) {
            auto pos = std::find_if(
                _operands.begin(),
                _operands.end(),
                [](const OperandInfo& op) {
                    return op._type == OperandTypes::Input;
                });
            _operands.emplace(pos, _out_tensor, OperandTypes::Output);
        };
        void add_input(Tensor& _in_tensor) {
            _operands.emplace_back(_in_tensor, OperandTypes::Input);
        };
        void add_input(const Tensor& _in_tensor) {
            _operands.emplace_back(_in_tensor, OperandTypes::Input);
        }
        void infer();
        void build();

        size_t numel() {
            return cpp20::compute_numel(_commo_shape);
        }
        size_t input_storage_offset(size_t indx = 0) {
            return _operands[_num_out + indx + 1]._storage_offset;
        }
        size_t input_storage_offset(size_t indx = 0) {
            return _operands[indx]._storage_offset;
        }
        uint8_t* output_ptr(size_t indx = 0) {
            return _operands[indx]._head_pointer;
        }
        uint8_t* input_ptr(size_t indx = 0) {
            return _operands[_num_out + indx + 1]._head_pointer;
        }

        std::tuple<std::vector<int64_t>, std::vector<int64_t>>
            input_size_and_stride(size_t indx = 0) {
                auto& op = _operands[_num_out + 1 + indx];
                return std::make_tuple(op._head_shape, op._head_stride);
            }
        std::tuple<std::vector<int64_t>, std::vector<int64_t>>
            output_size_and_stride(size_t indx = 0) {
                auto& op = _operands[indx];
                return std::make_tuple(op._head_shape, op._head_stride);
            }
        std::vector<int64_t> get_commo_shape() {
            return _commo_shape;
        }
        cpp20::Dtype get_out_dtype() {return out_dtype;}
    };
}
