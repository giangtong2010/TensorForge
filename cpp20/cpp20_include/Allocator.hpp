#pragma once
#include "_allocator.hpp"

namespace cpp20 {
    class Device;
    Allocator& get_allocator(Device device);
}