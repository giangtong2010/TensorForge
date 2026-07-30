#pragma once
#include "_tensor_iterator.hpp"
#include "xpu.hpp"
#include <tuple>
#include <sycl/sycl.hpp>

template <typename scalar_t, class Func>
void xpu_kernel(const at::TensorIterator& iter, Func&& op) {
    auto based_queue = xpu::based_queues::instance();
    const size_t numel = iter.numel();
    sycl::queue& q = based_queue.get_queue(iter.output_ptr());

    scalar_t* out_data = reinterpret_cast<scalar_t*>(iter.output_ptr());
    const scalar_t* a_data = reinterpret_cast<const scalar_t*>(iter.input_ptr(0));
    const scalar_t* b_data = reinterpret_cast<const scalar_t*>(iter.input_ptr(1));

    std::vector<int64_t> a_size = std::get<0>(iter.input_size_and_stride(0));
    std::vector<int64_t> a_stride = std::get<1>(iter.input_size_and_stride(0));
    std::vector<int64_t> b_size = std::get<0>(iter.input_size_and_stride(1));
    std::vector<int64_t> b_stride = std::get<1>(iter.input_size_and_stride(1));

    int64_t* a_size_dev = xpu::utils::malloc_shared_copy<int64_t>(
        a_size, a_size.size(), q
    );
    int64_t* a_stride_dev = xpu::utils::malloc_shared_copy<int64_t>(
        a_stride, a_stride.size(), q
    );
    int64_t* b_size_dev = xpu::utils::malloc_shared_copy<int64_t>(
        b_size, b_size.size(), q
    );
    int64_t* b_stride_dev = xpu::utils::malloc_shared_copy<int64_t>(
        b_stride, b_stride.size(), q
    );

    at::OffsetCalculator a_offset(
        iter.input_storage_offset(0),
        iter.get_input_ndim(0),
        a_size_dev,
        a_stride_dev
    );
    at::OffsetCalculator b_offset(
        iter.input_storage_offset(1),
        iter.get_input_ndim(1),
        b_size_dev,
        b_stride_dev
    );

    q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> id) {
        size_t linear_indx = id[0];
        size_t a_indx = a_offset(linear_indx);
        size_t b_indx = b_offset(linear_indx);

        out_data[linear_indx] = op(a_data[a_indx], b_data[b_indx]);
    });
}
