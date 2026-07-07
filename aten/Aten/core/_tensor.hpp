#pragma once
#include "impl/tensor_impl.hpp"
#include "intrusive_ptr.hpp"
#include <vector>

namespace at {
    class Tensor {
        cpp20::intrusive_ptr<TensorImpl> _impl;

    public:
        template <typename T>
        Tensor(
            std::initializer_list<T>,
            cpp20::Dtype,
            cpp20::Device,
            bool
        );
        ~Tensor() = default;

        // creation ops
        static Tensor empty(const std::vector<int64_t>&, const cpp20::Device);
        static Tensor zeros(const std::vector<int64_t>&, const cpp20::Device);
        static Tensor ones(const std::vector<int64_t>&, const cpp20::Device);
        static Tensor full(const std::vector<int64_t>&, const cpp20::Device);
        static Tensor arange(
            const size_t start, 
            const size_t end, 
            const size_t step,
            const cpp20::Dtype,
            const cpp20::Device
        );
        static Tensor rand(const std::vector<int64_t>&, const cpp20::Dtype, const cpp20::Device);

        // view ops
        Tensor view(const std::vector<int64_t>&) const;
        Tensor reshape(const std::vector<int64_t>&) const;
        Tensor transpose(int64_t, int64_t) const;
        Tensor permute(const std::vector<int64_t>&) const;
        Tensor squeeze(size_t) const;
        Tensor unsqueeze(size_t) const;
        void squeeze_(size_t) const;
        void unsqueeze_(size_t) const;

        // memory layout
        bool is_contiguous() const;
        Tensor contiguous() const;
        Tensor clone() const;
        Tensor detach() const;

        // arithmetic ops
        Tensor add(const Tensor&) const;
        Tensor sub(const Tensor&) const;
        Tensor mul(const Tensor&) const;
        Tensor div(const Tensor&) const;
        Tensor matmul(const Tensor&) const;

        // reduction ops
        Tensor sum() const;
        Tensor sum(const int64_t) const;
        Tensor sum(const std::initializer_list<int64_t>&) const;
        Tensor sum(const int64_t, bool) const;
        Tensor mean() const;
        Tensor max() const;
        Tensor min() const;

        // index ops
        Tensor index(const std::vector<int64_t>&) const;
        
        // shape ops
        Tensor flatten() const;
        Tensor flatten(int64_t start_dim, int64_t end_dim) const;
        Tensor expand(const std::vector<int64_t>&) const;
        Tensor repeat(const std::vector<int64_t>&) const;

        void override_size_and_stride(
            std::vector<int64_t>& new_size,
            std::vector<int64_t>& new_stride
        ) {_impl->ovr_size_and_stride(new_size, new_stride);}

        void override_impl(
            cpp20::intrusive_ptr<TensorImpl> new_impl
        ) {_impl = new_impl;}

        cpp20::Device get_device() noexcept {return _impl->_device;};
        const cpp20::Device get_device() const noexcept {return _impl->_device;};
        uint8_t* data() noexcept {return _impl->_storage.data();};
        const uint8_t* data() const noexcept {return _impl->_storage.data();};
        cpp20::intrusive_ptr<TensorImpl>& get_impl() noexcept {return _impl;};
        const cpp20::intrusive_ptr<TensorImpl>& get_impl() const noexcept {return _impl;};
        size_t get_numel() noexcept {return _impl->_numel;};
        const size_t get_numel() const noexcept {return _impl->_numel;};

        std::vector<int64_t>& get_size() noexcept {return _impl->_size;};
        const std::vector<int64_t>& get_size() const noexcept {return _impl->_size;};
        std::vector<int64_t>& get_stride() noexcept {return _impl->_stride;};
        const std::vector<int64_t>& get_stride() const noexcept {return _impl->_stride;};
    };
}
