#pragma once
#include "_dispatcher.hpp"
#include "_table.hpp"
#define REGISTER_KERNEL(op, backend, type, kernel)                             \
namespace {                                                                         \
    static bool reg_##op##_##backend = []() {                                       \
        dispatcher::Dispatcher::instance().register_kernel(                         \
            dispatcher::OP::op,                                                     \
            dispatcher::Backends::backend,                                          \
            dispatcher::Types::type                                                 \
            kernel                                                      \
        );                                                              \
        return true;                                                    \
    }();                                                                \
}                                                                       \
