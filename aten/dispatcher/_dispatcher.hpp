#pragma once
#include "_op.hpp"
#include "Dtype.hpp"
#include "Device.hpp"
#include <cstddef>
#include <vector>
#include <type_traits>

namespace at {
    class Tensor;
}

namespace dispatcher {
    using BinaryKernelFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&);
    using MemOpsKernelFn = at::Tensor (*)(at::Tensor& dst, const at::Tensor& src);
    using MemKernelFn = void (*)(void* dst, const void* src, cpp20::Dtype dtype, const size_t numel);

    template <typename Fn>
    class DispatchTable {
        Fn _table[(size_t) OP::count]
                [(size_t) cpp20::DeviceType::count]
                [(size_t) cpp20::Dtype::count] = {};

    public:
        void register_kernel(OP op, cpp20::DeviceType device, cpp20::Dtype dtype, Fn fn) {
            _table[(size_t) op][(size_t) device][(size_t) dtype] = fn;
        }
        Fn get_kernel(OP op, cpp20::DeviceType device, cpp20::Dtype dtype) {
            return _table[(size_t) op][(size_t) device][(size_t) dtype];
        }
    };

    class Dispatcher {
        DispatchTable<BinaryKernelFn> binary_;
        DispatchTable<MemOpsKernelFn> memops_;
        DispatchTable<MemKernelFn> memkn_;

    public:
        void register_kernel(
            OP op, 
            cpp20::DeviceType device,
            cpp20::Dtype dtype,
            BinaryKernelFn fn
        ) {
            binary_.register_kernel(op, device, dtype, fn);
        }
        void register_kernel(
            OP op,
            cpp20::DeviceType device,
            cpp20::Dtype dtype,
            MemOpsKernelFn fn
        ) {
            memops_.register_kernel(op, device, dtype, fn);
        }
        void register_kernel(
            OP op, 
            cpp20::DeviceType device,
            cpp20::Dtype dtype,
            MemKernelFn fn
        ) {
            memkn_.register_kernel(op, device, dtype, fn);
        }

        template <typename Fn>
        Fn get_kernel(OP op, cpp20::DeviceType device, cpp20::Dtype dtype) {
            if constexpr (std::is_same_v<Fn, BinaryKernelFn>) return binary_.get_kernel(op, device, dtype);
            else if constexpr (std::is_same_v<Fn, MemOpsKernelFn>) return memops_.get_kernel(op, device, dtype);
            else if constexpr (std::is_same_v<Fn, MemKernelFn>) return memkn_.get_kernel(op, device, dtype);
        }

        static Dispatcher& instance() {
            static Dispatcher dispatcher;
            return dispatcher;
        }
    };
}
