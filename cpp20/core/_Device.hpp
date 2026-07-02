#include <cstddef>

enum DeviceType {
    CPU,
    XPU
};
using DeviceIndx = size_t;

struct Device {
    DeviceType dev_type;
    DeviceIndx dev_indx;

    Device(DeviceType dev_type, DeviceIndx dev_idnx)
        : dev_type(dev_type), dev_indx(dev_indx) {}
    ~Device() = default;
};
