#pragma once
#include <cstddef>

namespace cpp20 {
    enum class DeviceType {
        CPU,
        XPU
    };
    using DeviceIndx = size_t;

    struct Device {
        DeviceType _dev_type;
        DeviceIndx _dev_indx;

        Device(DeviceType dev_type, DeviceIndx dev_idnx)
            : _dev_type(dev_type), _dev_indx(dev_idnx) {}
        ~Device() = default;
    };
}
