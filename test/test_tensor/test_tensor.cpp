#include "tensor.hpp"
#include <iostream>

int main() {
    at::Tensor tensor = at::Tensor(
        std::initializer_list<std::initializer_list<int>> {
            {1, 2, 3, 5, 6},
            {7, 8, 9, 10, 11}
        },
        cpp20::Dtype::Float32,
        cpp20::Device(cpp20::DeviceType::CPU, 0)
    );
    std::cout<< tensor;
}
