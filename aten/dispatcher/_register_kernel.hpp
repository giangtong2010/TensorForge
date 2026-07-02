#pragma once
#include "_dispatcher.hpp"
#define REGISTER_KERNEL(op, backend, kernel)                             \
namespace {                                                             \
    static bool reg_##op##_##backend = []() {                           \
        dispatcher::Dispatcher::instance().register_kernel(             \
            OP::op,                                                     \
            Backends::backend,                                          \
            kernel                                                      \
        );                                                              \
        return true;                                                    \
    }();                                                                \
}                                                                       \
