#include "backend.hpp"
#include "Allocator.hpp"
#include "Device.hpp"
#include <cstdint>
#include <iostream>

int main() {
    cpp20::Device device = cpp20::Device {cpp20::DeviceType::CPU, 0};
    auto allocator = cpp20::get_allocator(device);
    auto data_ptr = allocator.allocate(100000);
    uint8_t* ptr = data_ptr.get_data();

    std::cout <<"Địa chỉ bộ nhớ: " <<ptr;
}
