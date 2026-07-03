#include "_intrusive_refcounted.hpp"
#include <utility>

namespace cpp20 {
    struct AdoptRefcount {};
    inline constexpr AdoptRefcount adopt_refcount_t {};

    template <typename T>
    class intrusive_ptr {
        T* _ptr = nullptr;
    
    public:
        intrusive_ptr() = default;
        explicit intrusive_ptr(T* other, AdoptRefcount)
            : _ptr(other) {}
        explicit intrusive_ptr(T* other) : _ptr(other) {
            if (_ptr)
                _ptr->retain();
        };
    
        intrusive_ptr(const intrusive_ptr& other) noexcept
            : _ptr(other.get()) 
        {
            if (_ptr)
                _ptr->retain();
        };
        intrusive_ptr& operator=(const intrusive_ptr& other) noexcept {
            if (this != &other) {
                if (other._ptr)
                    other._ptr->retain();

                if (_ptr)
                    this->reset();
                _ptr = other._ptr;
            }
            return *this;
        };

        intrusive_ptr(intrusive_ptr&& other) noexcept
            : _ptr(other._ptr) {
                if (other._ptr)
                    other._ptr = nullptr;
            };
        intrusive_ptr& operator=(intrusive_ptr&& other) noexcept {
            if (this != &other) {
                if (_ptr)
                    this->reset();
                swap(other);
                other._ptr = nullptr;
            }
            return *this;
        };

        ~intrusive_ptr() noexcept {
            if (_ptr)
                _ptr->release();
        };

        T* get() const noexcept {return _ptr;};
        const T* get() const noexcept {return _ptr;};
        T& operator*() const noexcept {return *_ptr;};
        T* operator->() const noexcept {return _ptr;};
        explicit operator bool() const noexcept {return _ptr ? true : false;};

        void reset() noexcept {
            if (_ptr)
                _ptr->release();
            _ptr = nullptr;
        };
        void swap(intrusive_ptr& other) noexcept {
            using std::swap;
            swap(_ptr, other._ptr);
        };
    };
}
