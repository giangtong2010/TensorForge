#include "dispatcher/dispatcher.hpp"
#include "_Dtype.hpp"
#include "tensor.hpp"

auto& dispatchers = dispatcher::Dispatcher::instance();
class InitData {
    cpp20::Dtype a_dtype;
    cpp20::Dtype b_dtype;
    cpp20::Dtype out_dtype;

    cpp20::DeviceType a_dev_type;
    cpp20::DeviceType b_dev_type;

public:
    InitData(const at::Tensor& a, const at::Tensor& b)
        : a_dtype(a.get_dtype()),
        b_dtype(b.get_dtype()),
        a_dev_type(a.get_device()._dev_type),
        b_dev_type(b.get_device()._dev_type) {
            out_dtype = cpp20::promote_dtype(a_dtype, b_dtype);
        };

    cpp20::Dtype get_out_dtype() {return out_dtype;}
    cpp20::DeviceType get_device() {return a_dev_type;}
};

namespace at::impl {
    Tensor add(const Tensor& a, const Tensor& b) {
        InitData init = InitData(a, b);
        auto out_dtype = init.get_out_dtype();
        auto dev = init.get_device();

        dispatcher::KernelFn add_kernel =
            dispatchers.get_kernel(
                dispatcher::OP::add,
                dev,
                out_dtype
            );
        return add_kernel(a, b);
    }
    Tensor sub(const Tensor& a, const Tensor& b) {
        InitData init  = InitData(a, b);
        auto out_dtype = init.get_out_dtype();
        auto dev = init.get_device();

        dispatcher::KernelFn sub_kernel = 
            dispatchers.get_kernel(
                dispatcher::OP::sub,
                dev,
                out_dtype
            );
        return sub_kernel(a, b);
    }
    Tensor mul(const Tensor& a, const Tensor& b) {
        if (a.get_size().size() == 0 || b.get_size().size() == 0) {
            InitData init = InitData(a, b);
            auto out_dtype = init.get_out_dtype();
            auto dev = init.get_device();

            dispatcher::KernelFn mul_kernel = 
                dispatchers.get_kernel(
                    dispatcher::OP::mul,
                    dev,
                    out_dtype
                );
            return mul_kernel(a, b);
        }
        else {
            throw std::invalid_argument(
                "Argument for mul must have at least tensor is scalar, if you want " \
                "to mul 2 matrix, use matmul"
            );
        }
    }
    Tensor div(const Tensor& a, const Tensor& b) {
        InitData init = InitData(a, b);
        auto out_dtype = init.get_out_dtype();
        auto dev = init.get_device();

        dispatcher::KernelFn div_kernel = 
            dispatchers.get_kernel(
                dispatcher::OP::div,
                dev,
                out_dtype
            );
        return div_kernel(a, b);
    }
    Tensor matmul(const Tensor& a, const Tensor& b) {
        InitData init = InitData(a, b);
        auto out_dtype = init.get_out_dtype();
        auto dev = init.get_device();

        dispatcher::KernelFn matmul_kernel = 
            dispatchers.get_kernel(
                dispatcher::OP::matmul,
                dev,
                out_dtype
            );
        return matmul_kernel(a, b);
    }
}
