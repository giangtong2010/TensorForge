#include "xpu.hpp"
#include "Allocator.hpp"
#include "Device.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    cpp20::Device device(cpp20::DeviceType::XPU, 0);
    auto& allocator = cpp20::get_allocator(device);
    auto data_ptr = allocator.allocate(4 * 10);
    std::cout <<"Địa chỉ bộ nhớ: " 
              <<static_cast<void*>(data_ptr.get_data())
              <<std::endl;
}
