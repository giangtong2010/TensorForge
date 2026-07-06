#include "_tensor_ops.hpp"
#include "utils.hpp"
#include <vector>
#include <stdexcept>
#include <optional>
#include <cstdint>

namespace at {
    Tensor view(const std::vector<int64_t>& index, Tensor& tensor) {
        std::optional<size_t> indx_mines = std::nullopt;
        size_t know_numel = 1;
        std::vector<int64_t> new_size;
        std::vector<int64_t> new_stride;
        if (!tensor.is_contiguous())
            throw std::invalid_argument(
                "Tensor is not contiguous, so it can not be use for vew"
            );

        for (size_t i = 0; i < index.size(); i++) {
            auto elem = index[i];
            if (elem == -1) {
                if (indx_mines != std::nullopt) 
                    throw std::invalid_argument(
                        "Argument for view has too much mines arg (-1)"
                    );
                else indx_mines = i;
                new_size.insert(new_size.begin() + i, -1);
            }
            else if (elem == 0 || elem < -1) {
                throw std::invalid_argument(
                    "Argument must not have 0 or arg that smaller than -1"
                );
            }
            else {
                new_size.push_back(elem);
                know_numel *= elem;
            }
        }

        if (indx_mines != std::nullopt) {
            if (tensor.get_numel() % know_numel != 0) {
                throw std::runtime_error(
                    "Knowable numel can not div to tensor numel"
                );
            }
            new_size[indx_mines.value()] = tensor.get_numel() / know_numel;
        }
        else {
            if (know_numel != tensor.get_numel())
            throw std::runtime_error(
                "Numel of new stride doesn't match it in old tensor"
            );
        }

        new_stride = cpp20::compute_strides(new_size);
        auto old_impl = tensor.get_impl();
        auto new_impl =
            cpp20::make_intrusive<TensorImpl>(*old_impl);

        Tensor new_tensor = tensor;
        new_impl->ovr_size_and_stride(new_size, new_stride);
        new_tensor.override_impl(
            cpp20::intrusive_ptr(new_impl)
        );
        return new_tensor;
    }
}
