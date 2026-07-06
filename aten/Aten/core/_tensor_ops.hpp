#pragma once
#include "_tensor.hpp"
#include <vector>
#include <cstdint>

// TODO: HEADERS FOR MEMORY LAYTOU, ARITHMETICS OPS AND REDUCTION OPS
// INDEX OPS, SHAPE OPS

namespace at {
    //creation ops
    static Tensor empty(const std::vector<int64_t>&, cpp20::Device);
    static Tensor zeros(const std::vector<int64_t>&, cpp20::Device);
    static Tensor ones(const std::vector<int64_t>&, cpp20::Device);
    static Tensor arrange(
        const size_t start,
        const size_t end,
        const cpp20::Dtype,
        const cpp20::Device
    );
    static Tensor rand(const std::vector<int64_t>&, cpp20::Dtype, cpp20::Device);

    // view ops
    Tensor view(const std::vector<int64_t>&, Tensor& tensor);
    Tensor reshape(const std::vector<int64_t>&, Tensor& tensor);
    Tensor transpose(int64_t, int64_t, Tensor& tensor);
    Tensor permute(const std::vector<int64_t>&, Tensor& tensor);
    Tensor unsqueeze(size_t, Tensor& tensor);
    Tensor squeeze(size_t, Tensor& tensor);
    void squeeze_(size_t, Tensor& tensor);
    void unsqueeze_(size_t, Tensor& tensor);

    // memory layout
    bool is_contiguous(const Tensor&);
    Tensor contiguous(const Tensor&);
    Tensor clone(const Tensor&);
    Tensor detach(const Tensor&);

    // arithmetic ops
    Tensor add(const Tensor&, const Tensor&);
    Tensor sub(const Tensor&, const Tensor&);
    Tensor mul(const Tensor&, const Tensor&);
    Tensor div(const Tensor&, const Tensor&);
    Tensor matmul(const Tensor&, const Tensor&);
    
}