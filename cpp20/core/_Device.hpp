#pragma once
#include <stdexcept>

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