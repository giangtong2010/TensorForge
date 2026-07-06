#include "xpu.hpp"
#include "Allocator.hpp"
#include "Device.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    try {
        cpp20::Data_ptr data_ptr = cpp20::get_allocator(
            cpp20::Device(cpp20::DeviceType::XPU, 0)
        ).allocate(15);
        std::cout << data_ptr.get_data();
    } catch (const std::exception& e) {
        std::cerr << e.what();
    }
}
