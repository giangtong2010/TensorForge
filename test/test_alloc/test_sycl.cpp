#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    try {
        auto platforms = sycl::platform::get_platforms();

        std::cout << "Platforms: " << platforms.size() << '\n';

        for (auto &p : platforms)
            std::cout << p.get_info<sycl::info::platform::name>() << '\n';
    }
    catch (const std::exception &e) {
        std::cout << typeid(e).name() << '\n';
        std::cout << e.what() << '\n';
    }
}