#pragma once
#include "_dispatcher.hpp"
#include "_table.hpp"
#define REGISTER_KERNEL(op, backend, kernel)                             \
namespace {                                                                         \
    static bool reg_##op##_##backend = []() {                                       \
        dispatcher::Dispatcher::instance().register_kernel(                         \
            dispatcher::OP::op,                                                     \
            dispatcher::Backends::backend,                                          \
            kernel                                                      \
        );                                                              \
        return true;                                                    \
    }();                                                                \
}                                                                       \
