#include "xpu.hpp"
#include "Allocator.hpp"
#include "Device.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    std::cout <<"Start" <<std::endl;
    try {
        std::cout <<"In try" <<std::endl;
        
        std::cout <<"1" <<std::endl;
        auto& allocator = cpp20::get_allocator(
            cpp20::Device(cpp20::DeviceType::XPU, 0)
        );
        std::cout <<"2" <<std::endl;
        cpp20::Data_ptr data_ptr = allocator.allocate(4 * 10);
        std::cout <<"3" <<std::endl;
        std::cout << data_ptr.get_data();
        std::cout <<"4" <<std::endl;
    }
    catch (const std::exception& e) {
        std::cout <<"In catch" <<std::endl;
        std::cout <<e.what();
    }
    catch(...) {
        std::cout <<"Unknow error" <<std::endl;
    }
    std::cout <<"End process";
}
