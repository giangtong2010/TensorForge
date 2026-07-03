#include "impl/tensor_impl.hpp"
#include "_intrusive_ptr.hpp"
#include <vector>

namespace at {
    class Tensor {
        cpp20::intrusive_ptr<TensorImpl> _impl;

    public:
        template <typename T>
        Tensor(
            std::initializer_list<T>,
            cpp20::Dtype,
            Device,
            bool
        );
        ~Tensor() = default;

        // creation ops
        static Tensor empty(const std::vector<int64_t>&);
        static Tensor zeros(const std::vector<int64_t>&);
        static Tensor ones(const std::vector<int64_t>&);
        static Tensor full(const std::vector<int64_t>&);
        static Tensor arange(
            const size_t start, 
            const size_t end, 
            const size_t step,
            const cpp20::Dtype,
            const Device
        );
        static Tensor rand(const std::vector<int64_t>&, cpp20::Dtype, Device);

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

        cpp20::intrusive_ptr<TensorImpl>& get_impl() noexcept {return _impl;};
        const cpp20::intrusive_ptr<TensorImpl>& get_impl() const noexcept {return _impl;};

        std::vector<int64_t>& get_size() noexcept {return _impl->_size;};
        const std::vector<int64_t>& get_size() const noexcept {return _impl->_size;};
        std::vector<int64_t>& get_stride() noexcept {return _impl->_stride;};
        const std::vector<int64_t>& get_stride() const noexcept {return _impl->_stride;};
    };
}
