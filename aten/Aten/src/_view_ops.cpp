#include "tensor.hpp"
#include "utils.hpp"
#include <utility>
#include <vector>
#include <stdexcept>
#include <optional>
#include <cstdint>

namespace at::impl {
    Tensor view(const Tensor& tensor, const std::vector<int64_t>& index) {
        std::optional<size_t> indx_mines = std::nullopt;
        size_t know_numel = 1;
        std::vector<int64_t> new_size;
        std::vector<int64_t> new_stride;
        if (!tensor.is_contiguous())
            throw std::invalid_argument(
                "Tensor is not contiguous, so it can not be use for view"
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

    Tensor reshape(const Tensor& tensor, const std::vector<int64_t>& new_size) {
        if (tensor.is_contiguous())
            return at::impl::view(tensor, new_size);
        
        Tensor contiguous_tensor = tensor.contiguous();
        return at::impl::view(contiguous_tensor, new_size);
    }

    Tensor transpose(const Tensor& tensor, const int64_t dim1, const int64_t dim2) {
        size_t ndim = tensor.get_size().size();
        size_t indx1 = std::abs(dim1);
        size_t indx2 = std::abs(dim2);

        if (dim1 < 0)
            indx1 = ndim + dim1;
        if (dim2 < 0)
            indx2 = ndim + dim2;

        if (indx1 >= ndim || indx2 >= ndim)
            throw std::out_of_range(
                "transpose dimension out of range"
            );

        Tensor out_tensor = tensor;
        auto& out_size = out_tensor.get_size();
        auto& out_stride = out_tensor.get_stride();
        std::swap(out_size[indx1], out_size[indx2]);
        std::swap(out_stride[indx1], out_stride[indx2]);

        return out_tensor;
    }

    Tensor permute(const Tensor& tensor, const std::vector<int64_t>& perm) {
        if (perm.size() != tensor.get_size().size())
            throw std::invalid_argument(
                "perm.size() must equal tensor.ndim()"
            );

        if (cpp20::is_perm(perm, tensor.get_size())) {
            size_t ndim = tensor.get_size().size();
            Tensor out = tensor;

            for (size_t i = 0; i < ndim; i++) {
                size_t indx = std::abs(perm[i]);
                if (perm[i] < 0)
                    indx = ndim + perm[i];

                out.get_size()[i] = tensor.get_size()[indx];
                out.get_stride()[i] = tensor.get_stride()[indx];
            }

            return out;
        }
        else {
            throw std::invalid_argument(
                "List of indexs for permute was not permute, try to use the different once"
            );
        }
    }

    Tensor squeeze(const Tensor& tensor, const size_t indx) {
        Tensor out = tensor;
        squeeze_(out, indx);
        return out;
    }
    Tensor unsqueeze(const Tensor& tensor, const size_t indx) {
        Tensor out = tensor;
        unsqueeze_(out, indx);
        return out;
    }

    void squeeze_(Tensor& tensor, const size_t indx) {
        auto& tensor_size = tensor.get_size();
        auto& tensor_stride = tensor.get_stride();

        if (indx >= tensor_size.size())
            throw std::out_of_range(
                "index is ou of range (squeeze_)"
            );

        if (tensor_size[indx] == 1) {
            tensor_size.erase(tensor_size.begin() + indx);
            tensor_stride.erase(tensor_stride.begin() + indx);
        }
    }
    void unsqueeze_(Tensor& tensor, const size_t indx) {
        auto& tensor_size = tensor.get_size();
        auto& tensor_stride = tensor.get_stride();

        if (indx > tensor_size.size())
            throw std::out_of_range(
                "index is out of range (unsqueeze_)"
            );

        tensor_size.insert(tensor_size.begin() + indx, 1);
        tensor_stride = cpp20::compute_strides(tensor_size);
    }
}
