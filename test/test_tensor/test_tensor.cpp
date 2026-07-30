#include "tensor.hpp"
#include <vector>

int main() {
    auto device = cpp20::Device(
        cpp20::DeviceType::XPU,
        0
    );
    auto tensor = at::Tensor(
        std::initializer_list<std::initializer_list<int>> {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        },
        cpp20::Dtype::Float32,
        device,
        false
    );
}
