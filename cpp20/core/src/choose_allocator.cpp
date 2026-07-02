#include "Allocator.hpp"
#include "_Device.hpp"
#include "backend.hpp"
#include <stdexcept>

namespace cpp20{
    static Allocator& get_allocator(Device device) {
        switch (device.dev_type) {
           case DeviceType::CPU: return cpu::CPUAllocator::instance();
           case DeviceType::XPU: return xpu::XPUAllocator::instance();
            
           default:
                throw std::invalid_argument("Unsupported device");
        }
    }
}
