#include <thread>
#include <algorithm>

template <class function>
void parallel_for(
    size_t begin,
    size_t end,
    size_t grain_size,
    function&& func
) {
    const unsigned int num_threads =
        std::max(1u, std::thread::hardware_concurrency());
    
    size_t tasks = end - begin;
    int num_tasks = (tasks + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; t++) {
        size_t thread_begin = t * num_tasks;
        size_t thread_end = std::min(begin + num_tasks, end);

        if (thread_begin >= thread_end)
            break;

        func(begin, end);
    }
}
