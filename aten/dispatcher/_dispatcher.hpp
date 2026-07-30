#pragma once
#include "_table.hpp"
#include "Dtype.hpp"
#include "Device.hpp"
#include <cstddef>
#include <vector>

namespace at {
    class Tensor;
}

namespace dispatcher {
    using KernelFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&);
    using CopyKernelFn = void (*)(at::Tensor& dst, const at::Tensor& src);
    using MemcpyKernelFn = void (*)(void* dst, const void* src, cpp20::Dtype dtype, const size_t numel);

    class Dispatcher {
        KernelFn _table[(size_t) OP::count][(size_t) cpp20::Dtype::count][(size_t) cpp20::DeviceType::count] = {};
        CopyKernelFn _copyTable[(size_t) cpp20::Dtype::count][(size_t) cpp20::DeviceType::count] = {};
        MemcpyKernelFn _memcpyTable[(size_t) OP::count][(size_t) cpp20::Dtype::count][(size_t) cpp20::DeviceType::count] = {};

    public:
        static Dispatcher& instance() {
            static Dispatcher dispatcher;
            return dispatcher;
        }

        void register_kernel(
            OP op, cpp20::DeviceType backend, cpp20::Dtype type, KernelFn fn
        ) {
            _table[(size_t) op][(size_t) type][(size_t) backend] = fn;
        }
        void register_copy_kernel(
            cpp20::DeviceType backend, cpp20::Dtype type, CopyKernelFn fn
        ) {
            _copyTable[(size_t) type][(size_t) backend] = fn;
        }

        void register_memcpy_kernel(
            OP op, cpp20::DeviceType backend, cpp20::Dtype type, MemcpyKernelFn fn
        ) {
            _memcpyTable[(size_t) op][(size_t) type][(size_t) backend] = fn;
        }

        CopyKernelFn get_copy_kernel(cpp20::DeviceType backend, cpp20::Dtype type) {
            return _copyTable[(size_t) type][(size_t) backend];
        }
        KernelFn get_kernel(OP op, cpp20::DeviceType backend, cpp20::Dtype type) {
            return _table[(size_t) op][(size_t) type][(size_t) backend];
        }
        MemcpyKernelFn get_memcpy_kernel(OP op, cpp20::DeviceType backend, cpp20::Dtype type) {
            return _memcpyTable[(size_t) op][(size_t) type][(size_t) backend];
        }
    };
}
