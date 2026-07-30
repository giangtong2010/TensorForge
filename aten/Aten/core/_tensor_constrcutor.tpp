#include "_storage.hpp"
#include "Allocator.hpp"
#include "dispatcher/dispatcher.hpp"

#include <stack>
#include <vector>
#include <type_traits>
#include <tuple>

template <typename T>
struct is_initializer_list : std::false_type {};
template <typename T>
struct is_initializer_list<std::initializer_list<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_initializer_list_v = is_initializer_list<T>::value;

template <typename T>
struct leaf_type {
    using type = T;
};
template <typename T>
struct leaf_type<std::initializer_list<T>> {
    using type = typename leaf_type<T>::type;
};


template <typename T>
void _infer_dtype_and_size(
    const std::initializer_list<T>& list, std::vector<int64_t>& size, 
    std::vector<typename leaf_type<T>::type>& flatten_data,
    cpp20::Dtype& dtype, size_t depth, bool& first
) {
    int64_t current_size = list.size();
    if (depth == size.size()) {
            size.push_back(current_size);
    }
    else {
        if (size[depth] != current_size) {
            throw std::runtime_error(
                std::format(
                    "Inconsistent tensor shape at dimension {}: expected {}, got {}",
                    depth, size[depth], current_size
                )
            );
        }
    }

    if constexpr (is_initializer_list_v<T>) {
        for (const auto inp : list) {
            _infer_dtype_and_size(inp, size, flatten_data, dtype, depth + 1, first);
        }
    }
    else {
        using _type = typename leaf_type<T>::type;
        cpp20::Dtype current_dtype = cpp20::CPPTypeToDtype<_type>::value;

        if (first) {
            dtype = current_dtype;
            first = false;
        }
        else {
            dtype = cpp20::promote_dtype(dtype, current_dtype);
        }

        flatten_data.insert(
            flatten_data.end(),
            list.begin(),
            list.end()
        );
    }
}

template <typename T>
struct _res {
    cpp20::Dtype dtype;
    std::vector<int64_t> size;
    std::vector<typename leaf_type<T>::type> flatten_data;

    _res() = default;
};

template <typename T>
_res<T> _infer(const std::initializer_list<T>& list) {
    _res<T> res;
    bool first = true;
    _infer_dtype_and_size(
        list, 
        res.size, 
        res.flatten_data, 
        res.dtype, 
        0, 
        first
    );
    return res;
}

namespace at {
    // TODO: WRITE CAST_KERNEL TO HELP CAST
    // EXAMPLE:
    //  int and cpp20::Dtype::Float32
    //  cast int -> float
    
    template <typename T>
    cpp20::intrusive_ptr<TensorImpl> init(
        std::initializer_list<T> list, std::vector<int64_t>& size, 
        std::vector<typename leaf_type<T>::type>& flatten_data, 
        cpp20::Dtype dtype, cpp20::Device device
    ) {
        auto& allocator = cpp20::get_allocator(device);
        size_t numel = cpp20::compute_numel(size);
        cpp20::Data_ptr data_ptr = allocator.allocate(
            numel * cpp20::dtype_size(dtype)
        );

        auto storage_impl = cpp20::make_intrusive<StorageImpl>(
            std::move(data_ptr),
            numel * cpp20::dtype_size(dtype),
            device,
            &allocator
        );
        Storage storage(storage_impl);
        auto tensor_impl = cpp20::make_intrusive<TensorImpl>(
            storage,
            size,
            dtype,
            device,
            0
        );

        dispatcher::MemcpyKernelFn copy_from_host_kernel =
            dispatcher::Dispatcher::instance().get_memcpy_kernel(
                dispatcher::OP::copy_from_host, device._dev_type, dtype
            );
        
        copy_from_host_kernel(data_ptr.get_data(), flatten_data.data(), dtype, flatten_data.size());
        return tensor_impl;
    }

    template <typename T>
    Tensor::Tensor(
        std::initializer_list<T> data, 
        cpp20::Dtype dtype, 
        cpp20::Device device, 
        bool requires_grad
    ) {
        auto res = _infer<T>(data);
        auto tensor_impl = init<T>(data, res.size, res.flatten_data, dtype, device);
        _impl = tensor_impl;
    }

    template <typename T>
    Tensor::Tensor(
        std::initializer_list<T> data,
        cpp20::Device device,
        bool requires_grad
    ) {
        auto res = _infer<T>(data);
        _impl = 
            init<T>(data, res.size, res.flatten_data, res.dtype, device);
    }

    template <typename T>
    Tensor::Tensor(
        std::initializer_list<T> data,
        bool requires_grad
    ) {
        cpp20::Device device = cpp20::Device(
            cpp20::DeviceType::CPU,
            0
        );
        auto res = _infer<T>(data);
        _impl = 
            init<T>(data, res.size, res.flatten_data, res.dtype, device);
    }

    template <typename T>
    Tensor::Tensor(
        std::initializer_list<T> data,
        cpp20::Dtype dtype,
        bool requires_grad
    ) {
        cpp20::Device device = cpp20::Device(
            cpp20::DeviceType::CPU,
            0
        );
        auto res = _infer<T>(data);
        _impl = 
            init<T>(data, res.size, res.flatten_data, dtype, device);
    }
}
