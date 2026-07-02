#include "intrusive_ptr/include/_intrusive_ptr.hpp"
#include "Allocator.hpp"
#include "Dtype.hpp"
#include "Device.hpp"
#include <utility>

namespace at {
    class StorageImpl : public cpp20::IntrusiveRefcounted<StorageImpl> {
        cpp20::Data_ptr _data_ptr;
        size_t _nbytes;
        Device _device;
        const cpp20::Allocator* _allocator;

    public:
        StorageImpl(
            cpp20::Data_ptr data, 
            size_t nbytes, 
            Device device, 
            cpp20::Allocator* allocator
        )
            : _data_ptr(std::move(data)),
            _nbytes(nbytes),
            _device(device),
            _allocator(std::move(allocator)) {};

        ~StorageImpl() = default;

        StorageImpl(const StorageImpl&) = delete;
        StorageImpl& operator=(const StorageImpl&) = delete;
        StorageImpl(StorageImpl&&) = delete;
        StorageImpl& operator=(StorageImpl&&) = delete;

        uint8_t* get_data() {return _data_ptr.get_data();};
        cpp20::Data_ptr& get_data_ptr() {return _data_ptr;};
    };
}
