#include "tensor.hpp"
#include "dispatcher/dispatcher.hpp"
#include "_storage_impl.hpp"
#include "_storage.hpp"
#include <optional>

namespace at::impl {
    bool is_contiguous(const Tensor& tensor) noexcept {
        const std::vector<int64_t>& size = tensor.get_size();
        const std::vector<int64_t>& stride = tensor.get_stride();

        if (stride.size() != size.size()) return false;
        if (size.empty()) return true;

        size_t last_dim = stride.size() - 1;

        if (stride[last_dim] != 1) return false;

        for (size_t i = last_dim; i-- > 0;) {
            size_t stride_i = stride[i + 1] * size[i + 1];
            if (stride[i] != stride_i) return false;
        }
        return true;
    }

    Tensor contiguous(const Tensor& tensor) noexcept {
        if (tensor.is_contiguous())
            return tensor;
        
            Tensor out = tensor.clone();

        dispatcher::MemOpsKernelFn contiguous =
            dispatcher::Dispatcher::instance().get_kernel<dispatcher::MemOpsKernelFn>(
                dispatcher::OP::contiguous,
                tensor.get_device()._dev_type,
                tensor.get_dtype()
            );

        return contiguous(out, tensor);
    }

    Tensor clone(const Tensor& tensor) noexcept {
        auto& allocator = cpp20::get_allocator(tensor.get_device());
        cpp20::Data_ptr out_data_ptr = allocator.allocate(
            tensor.get_numel() * cpp20::dtype_size(tensor.get_dtype())
        );
       
        auto storage_impl = 
            cpp20::make_intrusive<StorageImpl>(
                std::move(out_data_ptr),
                tensor.get_numel() * cpp20::dtype_size(tensor.get_dtype()),
                tensor.get_device(),
                &allocator
            );
        Storage storage(storage_impl);
        auto tensor_impl =
            cpp20::make_intrusive<TensorImpl>(
                storage,
                tensor.get_size(),
                tensor.get_stride(),
                tensor.get_dtype(),
                tensor.get_device(),
                tensor.get_storage_offset()
            );

        Tensor out(tensor_impl);
        dispatcher::MemOpsKernelFn copy_kernel = 
            dispatcher::Dispatcher::instance().get_kernel<dispatcher::MemOpsKernelFn>(
                dispatcher::OP::copy,
                tensor.get_device()._dev_type,
                tensor.get_dtype()
            );
        copy_kernel(out, tensor);

        return out;
    }
}
