#include "_storage_impl.hpp"
#include "_intrusive_ptr.hpp"

namespace at {
    struct Storage {
        cpp20::intrusive_ptr<StorageImpl> _impl;

        uint8_t* data() noexcept {return _impl.get()->get_data();}
        uint8_t* data() const noexcept {return _impl.get()->get_data();}
    };
}
