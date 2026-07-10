#pragma once
#include "tensor.hpp"
#include "_table.hpp"
#include "Dtype.hpp"
#include "Device.hpp"
#include <cstddef>
#include <vector>

namespace dispatcher {
    using KernelFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&);

    class Dispatcher {
        KernelFn _table[(size_t) OP::count][(size_t) cpp20::Dtype::count][(size_t) cpp20::DeviceType::count];

    public:
        static Dispatcher& instance() {
            static Dispatcher dispatcher;
            return dispatcher;
        }

        void register_kernel(
            OP op, cpp20::DeviceType backend, cpp20::Dtype type, KernelFn fn
        ) {
            _table[(size_t) op][(size_t) backend][(size_t) type] = fn;
        }

        KernelFn get_kernel(OP op, cpp20::DeviceType backend, cpp20::Dtype type) {
            return _table[(size_t) op][(size_t) backend][(size_t) type];
        }
    };
}
