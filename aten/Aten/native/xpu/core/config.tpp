#pragma once
#include "_tensor_iterator.hpp"
#include "xpu.hpp"
#include <tuple>
#include <sycl/sycl.hpp>

template <typename scalar_t, class Func>
void xpu_kernel(const at::TensorIterator& iter, Func&& op) {
    scalar_t* out_data = reinterpret_cast<scalar_t*>(iter.output_ptr());
    const scalar_t* a_data = reinterpret_cast<const scalar_t*>(iter.input_ptr(0));
    const scalar_t* b_data = reinterpret_cast<const scalar_t*>(iter.input_ptr(1));

    at::OffsetCalculator a_offset(
        iter.input_storage_offset(0),
        std::get<0>(iter.input_size_and_stride(0)),
        std::get<1>(iter.input_size_and_stride(0))
    );
    at::OffsetCalculator b_offset(
        iter.input_storage_offset(1),
        std::get<0>(iter.input_size_and_stride(1)),
        std::get<1>(iter.input_size_and_stride(1))
    );

    auto based_queue = xpu::based_queues::instance();
    const size_t numel = iter.numel();
    sycl::queue& q = based_queue.get_queue(iter.output_ptr());

    q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> id) {
        size_t linear_indx = id[0];
        size_t a_indx = a_offset(linear_indx);
        size_t b_indx = b_offset(linear_indx);

        out_data[linear_indx] = op(a_data[a_indx], b_data[b_indx]);
    });
}
