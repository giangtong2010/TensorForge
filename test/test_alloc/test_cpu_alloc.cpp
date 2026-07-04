#include "cpu.hpp"
#include "Allocator.hpp"
#include "Device.hpp"
#include <cstdint>
#include <iostream>

int main() {
    cpp20::Device device = cpp20::Device {cpp20::DeviceType::CPU, 0};
    auto data_ptr = cpp20::get_allocator(device).allocate(4 * 1000 * 1000);
    uint8_t* ptr = data_ptr.get_data();

    std::cout <<"Địa chỉ bộ nhớ: " <<ptr;
}
