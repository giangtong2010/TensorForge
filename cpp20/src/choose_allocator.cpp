#include "Allocator.hpp"
#include "Device.hpp"
#include "backend.hpp"
#include <stdexcept>

namespace cpp20{
    static Allocator& get_allocator(cpp20::Device device) {
        switch (device._dev_type) {
           case DeviceType::CPU: return cpu::CPUAllocator::instance();
           case DeviceType::XPU: return xpu::XPUAllocator::instance();
            
           default:
                throw std::invalid_argument("Unsupported device");
        }
    }
}
