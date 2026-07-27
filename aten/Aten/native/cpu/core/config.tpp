#include <thread>
#include <algorithm>
#include "_tensor_iterator.hpp"

template <class function>
void parallel_for(
    size_t begin,
    size_t end,
    size_t grain_size,
    function&& func
) {
    const unsigned int num_threads =
        std::max(1u, std::thread::hardware_concurrency());
    
    std::vector<std::thread> threads;
    
    size_t tasks = end - begin;
    int num_tasks = (tasks + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; t++) {
        size_t thread_begin = begin + t * num_tasks;
        size_t thread_end = std::min(thread_begin + num_tasks, end);

        if (thread_begin >= thread_end)
            break;

        threads.emplace_back([&, thread_begin, thread_end]() {
                func(thread_begin, thread_end);
            });
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

template <typename scalar_t, class Func>
void cpu_kernel(const at::TensorIterator& iter, Func&& op) {
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

    parallel_for(
        0,
        iter.numel(),
        1024,
        [&](size_t begin, size_t end) {
            for (size_t linear_indx = begin; linear_indx < end; linear_indx++) {
                size_t a_indx = a_offset(linear_indx);
                size_t b_indx = b_offset(linear_indx);

                out_data[linear_indx] = op(a_data[a_indx], b_data[b_indx]);
            }
        }
    );
}
