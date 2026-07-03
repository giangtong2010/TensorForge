#pragma once
#include <atomic>

namespace cpp20 {
    template <class Derived>
    class IntrusiveRefcounted {
        std::atomic<uint32_t> refcount = 1;
    
    public:
        void retain() noexcept {
            refcount.fetch_add(1, std::memory_order_relaxed);
        };
        void release() noexcept {
            refcount.fetch_sub(1, std::memory_order_acq_rel);
            if (refcount == 0) {
                delete static_cast<Derived*>(this);
            }
        };

        IntrusiveRefcounted(const IntrusiveRefcounted&) = delete;
        IntrusiveRefcounted& operator=(const IntrusiveRefcounted&) = delete;
        IntrusiveRefcounted(IntrusiveRefcounted&&) = delete;
        IntrusiveRefcounted& operator=(IntrusiveRefcounted&&) = delete;

    protected:
        ~IntrusiveRefcounted() = default;
        IntrusiveRefcounted() = default;
    };
}
