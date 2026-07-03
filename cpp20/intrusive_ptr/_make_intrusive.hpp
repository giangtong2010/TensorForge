#pragma once
#include "_intrusive_ptr.hpp"
#include <utility>

namespace cpp20 {
    template <typename T, typename... Arg>
    intrusive_ptr<T> make_intrusive(Arg&&... args) noexcept {
        return intrusive_ptr<T>(
            new T(std::forward<Arg>(args)...),
            adopt_refcount_t
        );
    }
}
