#pragma once
#include "_allocator.hpp"

namespace cpp20 {
    class Device;
    static Allocator& get_allocator(Device device);
}