#pragma once
#include "Aten/include/tensor.hpp"
#include "_table.hpp"
#include <cstddef>

namespace dispatcher {
    using KernelFn = void(*)(const at::Tensor&, const at::Tensor&, at::Tensor&);

    class Dispatcher {
        KernelFn _table[(size_t) OP::count][(size_t) Backends::count] {};

    public:
        static Dispatcher& instance() {
            static Dispatcher dispatcher;
            return dispatcher;
        }

        void register_kernel(
            OP op, Backends backend, KernelFn fn
        ) {
            _table[(size_t) op][(size_t) backend] = fn;
        }
    };
}
