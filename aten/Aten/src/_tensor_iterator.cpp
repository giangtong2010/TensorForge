#include <cstddef>
#include <stdexcept>
#include <optional>
#include <format>
#include <algorithm>
#include <string>
#include <cmath>

#include "_tensor_iterator.hpp"

namespace at {
    void TensorIterator::validate_operands() {
        _num_in = _num_out = 0;

        for (size_t indx = 0; indx < _operands.size(); indx++) {
            auto op = _operands[indx];
            if (op._type == OperandTypes::Input)
                _num_in++;
            else if (op._type == OperandTypes::Output)
                _num_out++;
        }

        if (_num_in == 0 && _num_out == 0)
            throw std::runtime_error(
                "tensor iterator do not have input and output op, use add_input(tensor) to add input op" \
                " and add_output(tensor) to add output op"
            );

        if (_num_in == 0)
            throw std::runtime_error(
                "tensor iterator do not have any input tensor, using add_input(tensor) to add input op"
            );
        if (_num_out == 0)
            throw std::runtime_error(
                "tensor iterator do not have any output tensor, using add_output(tensor) to add ouput op"
            );
    }
    void TensorIterator::coalesce_dimensions() {
        for (size_t i = 0; i < _operands.size(); i++) {
            OperandInfo& op = _operands[i];
            Tensor head_tensor = op._head_tensor;

            if (head_tensor.is_contiguous()) {
                op._head_shape = 
                    std::vector<int64_t>{static_cast<int64_t> (head_tensor.get_numel())};
                op._head_stride = std::vector<int64_t>{1};
            }
        }
    }
    void TensorIterator::broad_cast_shape() {
        size_t max_len_shape = 0;
        for (auto& op : _operands) {
            if (op._head_shape.size() > max_len_shape)
                max_len_shape = op._head_shape.size();
        }

        _commo_shape = std::vector<int64_t>(max_len_shape, 1);
        size_t ndim = _commo_shape.size();
        for (auto& op : _operands) {
            auto& tensor = op._head_tensor;
            size_t offset = ndim - op._head_shape.size();

            std::vector<int64_t> aliged_size(ndim, 1);

            std::copy(
                op._head_shape.begin(),
                op._head_shape.end(),
                aliged_size.begin() + offset
            );

            op._head_shape.swap(aliged_size);

            for (size_t i = 0; i < ndim; i++) {
                if (_commo_shape[i] == 1) {
                    _commo_shape[i] = std::max(_commo_shape[i], op._head_shape[i]);
                }
                else if (
                    op._head_shape[i] != 1 && 
                    _commo_shape[i] != op._head_shape[i]
                ){
                    throw std::runtime_error(
                        std::format(
                            "The size of tensor a ({}) must match the size of " \
                            "tensor b ({}) at non-singleton dimension {}",
                            op._head_shape[i], _commo_shape[i], i
                        )
                    );
                }
            }
        }
    }

    void TensorIterator::broad_cast_stride() {
        size_t ndim = _commo_shape.size();
        for (auto& op : _operands) {
            auto& tensor = op._head_tensor;
            size_t offset = ndim - op._head_shape.size();

            std::vector<int64_t> aliged_stride(ndim, 0);

            std::copy(
                op._head_stride.begin(),
                op._head_stride.end(),
                aliged_stride.begin() + offset
            );

            op._head_stride.swap(aliged_stride);

            for (size_t k = 0; k < ndim; k++) {
                if (op._head_shape[k] == 1 && _commo_shape[k] != 1)
                    op._head_stride[k] = 0;
                op._head_shape[k] = _commo_shape[k];
            }
        }
    }
    void TensorIterator::coalesce_dtype() {
        bool first = true;
        for (auto& op : _operands) {
            if (op._type == OperandTypes::Output)
                continue;

            if (first) {
                out_dtype = op._head_dtype;
                first = false;
            }
            else {
                out_dtype = cpp20::promote_dtype(
                    out_dtype,
                    op._head_dtype
                );
            }
        }
    }
    void TensorIterator::check_dev() {
        bool first = true;
        for (auto& op : _operands) {
            if (first) {
                out_device = op._head_device;
                first = false;
            } else {
                if (
                    out_device._dev_type != op._head_device._dev_type ||
                    out_device._dev_indx != op._head_device._dev_indx
                )
                    throw std::runtime_error(
                        std::format(
                            "Expected all tensors to be on the same device, but " \
                            "found at least two devices, {} and {}:{}!",
                            out_device._dev_type, op._head_device._dev_type, op._head_device._dev_indx
                        )
                    );
            }
        }
    }

    void TensorIterator::infer() {
        validate_operands();
        check_dev();
        broad_cast_shape();
        coalesce_dtype();
    }

    void TensorIterator::build() {
        validate_operands();
        coalesce_dimensions();
        broad_cast_stride();
    }
}
