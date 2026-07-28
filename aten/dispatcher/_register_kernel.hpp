#pragma once
#include "_dispatcher.hpp"
#include "_table.hpp"
#include "Dtype.hpp"
#include "Device.hpp"

#define REGISTER_KERNEL(op, backend, type, kernel)                              \
namespace {                                                                     \
    static bool reg_##op##_##backend##_##type = []() {                          \
        dispatcher::Dispatcher::instance().register_kernel(                     \
            dispatcher::OP::op,                                                 \
            cpp20::DeviceType::backend,                                         \
            cpp20::Dtype::type,                                                 \
            kernel                                                              \
        );                                                                      \
        return true;                                                            \
    }();                                                                        \
}                                                                               \
