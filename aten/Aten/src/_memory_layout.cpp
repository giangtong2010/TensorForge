#include "tensor.hpp"
#include "dispatcher/dispatcher.hpp"
#include <optional>

namespace at {
    bool is_contiguous(const Tensor& tensor) noexcept {
        const std::vector<int64_t>& size = tensor.get_size();
        const std::vector<int64_t>& stride = tensor.get_stride();
        size_t last_dim = stride.size() - 1;

        if (stride[last_dim] != 1) return false;
        if (stride.size() != size.size()) return false;
        for (size_t i = last_dim - 2; i-- > 0;) {
            size_t stride_i = stride[i + 1] * size[i + 1];
            if (stride[i] != stride_i) return false;
        }
        return true;
    }

    Tensor contiguous(const Tensor& tensor) noexcept {
        if (tensor.is_contiguous())
            return tensor;
            
        dispatcher::TypesAndBackends type_and_backend = 
            dispatcher::Map::instance().get_type_and_backend(
                tensor.get_dtype(), tensor.get_device()
            );

        dispatcher::KernelFn contiguous =
            dispatcher::Dispatcher::instance().get_kernel(
                dispatcher::OP::contiguous,
                type_and_backend.backend,
                type_and_backend.type
            );

        return contiguous(tensor, tensor);
    }
}
