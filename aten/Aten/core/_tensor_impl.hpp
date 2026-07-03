#pragma once
#include "impl/storage.hpp"
#include "utils.hpp"
#include "intrusive_ptr.hpp"
#include <vector>
#include <cstdint>
#include <utility>

namespace at {
    class TensorImpl : public cpp20::IntrusiveRefcounted<TensorImpl> {
        Storage _storage;
        std::vector<int64_t> _size = {};
        std::vector<int64_t> _stride = {};

        cpp20::Dtype _dtype;
        cpp20::Device _device;
        size_t _storage_offset = 0;
        size_t _numel = 0;

    protected:
        TensorImpl(
            Storage storage,
            std::vector<int64_t> size,
            cpp20::Dtype dtype,
            cpp20::Device device,
            size_t storage_offset
        )
            : _storage(std::move(storage)),
            _size(std::move(size)),
            _stride(cpp20::compute_strides(size)),
            _dtype(dtype),
            _device(device),
            _storage_offset(storage_offset),
            _numel(cpp20::compute_numel(size)) {};

        ~TensorImpl() = default;

        // allow move but not for copy
        TensorImpl(TensorImpl&& other) noexcept
            : _storage(std::move(other._storage)),
            _size(std::move(other._size)),
            _stride(std::move(other._stride)),
            _dtype(std::move(other._dtype)),
            _device(std::move(other._device)),
            _storage_offset(std::move(other._storage_offset)),
            _numel(std::move(other._numel)) {};

        TensorImpl& operator=(TensorImpl&& other) noexcept {
            if (this != &other) {
                _storage = std::move(other._storage);
                _size = std::move(other._size);
                _stride = std::move(other._stride);
                _dtype = std::move(other._dtype);
                _device = std::move(other._device);
                _storage_offset = std::move(other._storage_offset);
                _numel = std::move(other._numel);
            }
            return *this;
        };
        TensorImpl(const TensorImpl&) = delete;
        TensorImpl& operator=(const TensorImpl&) = delete;

        // operator
        void swap(TensorImpl& other) noexcept {
            using std::swap;
            swap(_storage, other._storage);
            swap(_dtype, other._dtype);
            swap(_device, other._device);
            swap(_size, other._size);
            swap(_stride, other._stride);
            swap(_numel, other._numel);
            swap(_storage_offset, other._storage_offset);
        }
        
        friend class Tensor;
    };
}
