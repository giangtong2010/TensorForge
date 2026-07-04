#pragma once
#include <cstdint>
#include <utility>

using Deleter = void (*)(uint8_t* data, void* ctx) noexcept;
namespace cpp20 {
    class Data_ptr {
        uint8_t* _data = nullptr;
        void* _ctx = nullptr;
        Deleter _deleter = nullptr;

        void move_from(Data_ptr&& other) {
            _data = other.get_data();
            _ctx = other.get_ctx();
            _deleter = other.get_deleter();

            other._data = nullptr;
            other._ctx = nullptr;
            other._deleter = nullptr;
        }

    public:
        Data_ptr() = default;
        Data_ptr(uint8_t* data, void* ctx, Deleter deleterfn)
            : _data(data), _ctx(ctx), _deleter(deleterfn) {}
        
        ~Data_ptr() {
            reset();
        }

        Data_ptr(const Data_ptr& other) = delete;
        Data_ptr& operator=(const Data_ptr& other) = delete;

        Data_ptr(Data_ptr&& other) noexcept {
            move_from(std::move(other));
        }
        Data_ptr& operator=(Data_ptr&& other) noexcept {
            if (this != &other) {
                reset();
                move_from(std::move(other));
            }
            return *this;
        }

        void reset() {
            if (_deleter && (_data || _ctx)) {
                _deleter(_data, _ctx);
            }
            _data = nullptr;
            _ctx = nullptr;
            _deleter = nullptr;
        }

        uint8_t* get_data() {return _data;}
        const uint8_t* get_data() const {return _data;};
        void* get_ctx() {return _ctx;}
        Deleter get_deleter() {return _deleter;}
    };

    struct Allocator {
    virtual cpp20::Data_ptr allocate(size_t nbytes) = 0;

    protected:
        Allocator() = default;
    };
}
