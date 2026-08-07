#include "tensor.hpp"

int main() {
    at::Tensor a(
        std::initializer_list<std::initializer_list<int>>
        {
            {1, 2, 3, 4},
            {5, 6, 7, 8}
        },
        cpp20::Dtype::Int32,
        cpp20::Device{cpp20::DeviceType::XPU, 0}
    );
    at::Tensor b(
        std::initializer_list<std::initializer_list<int>>
        {
            {1, 2, 3, 4},
            {5, 6, 7, 8}
        },
        cpp20::Dtype::Int32,
        cpp20::Device{cpp20::DeviceType::XPU, 0}
    );

    at::Tensor c = at::impl::add(a, b);

    std::cout <<"Tensor a: " <<a;
    std::cout <<"Tensor b: " <<b;
    std::cout <<"a+b: " <<c;
}
