#include "tensor.hpp"
#include "Allocator.hpp"
#include "utils.hpp"
#include "intrusive_ptr.hpp"
#include "_storage_impl.hpp"
#include "_tensor_impl.hpp"

namespace at::impl {
    Tensor empty(const std::vector<int64_t>& size, cpp20::Dtype dtype, cpp20::Device device) {
        size_t numel = cpp20::compute_numel(size);

        auto& allocator = cpp20::get_allocator(device);
        cpp20::Data_ptr data_ptr = 
            allocator.allocate(numel * cpp20::dtype_size(dtype));

        auto storage_impl = 
            cpp20::make_intrusive<StorageImpl>(
                std::move(data_ptr),
                numel * cpp20::dtype_size(dtype),
                device,
                &allocator
            );
        Storage storage(storage_impl);
        auto tensor_impl = 
            cpp20::make_intrusive<TensorImpl>(
                storage,
                size,
                dtype,
                device,
                0
            );
        
        return Tensor(tensor_impl);
    }
}