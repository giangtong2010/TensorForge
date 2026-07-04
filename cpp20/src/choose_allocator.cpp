#include "Allocator.hpp"
#include "Device.hpp"
#include "cpu.hpp"
#include "xpu.hpp"
#include <stdexcept>

namespace cpp20{
    Allocator& get_allocator(cpp20::Device device) {
        switch (device._dev_type) {
           case DeviceType::CPU: return cpu::CPUAllocator::instance();
           case DeviceType::XPU: return xpu::XPUAllocator::instance();
            
           default:
                throw std::invalid_argument("Unsupported device");
        }
    }
}
