#pragma once
#include <stdexcept>
#include <format>
#include <utility>

#if defined(__clang__) || defined(__GNUC__)
    #define TF_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
    #define TF_UNREACHABLE() __assume(false)
#else
    #include <cstdlib>
    #define TF_UNREACHABLE() std::abort()
#endif

namespace cpp20 {
    using DeviceIndx = size_t;
    enum class DeviceType {
        
        CPU,
        
        XPU,
        
        count=2,
    };

    struct Device {
        DeviceType _dev_type;
        DeviceIndx _dev_indx;

        Device(DeviceType dev_type, DeviceIndx dev_idnx)
            : _dev_type(dev_type), _dev_indx(dev_idnx) {}
        Device() = default;
        ~Device() = default;
    };
}
template <>
struct std::formatter<cpp20::DeviceType> : std::formatter<std::string_view> {
    auto format(
        cpp20::DeviceType dev,
        std::format_context& ctx
    ) const {
        std::string_view name = "Unknow";

        switch (dev) {
            
            case cpp20::DeviceType::CPU:
                name = "CPU";
                break;
            
            case cpp20::DeviceType::XPU:
                name = "XPU";
                break;
            
            case cpp20::DeviceType::count:
                TF_UNREACHABLE();
        }

        return std::formatter<std::string_view>::format(name, ctx);
    }
};