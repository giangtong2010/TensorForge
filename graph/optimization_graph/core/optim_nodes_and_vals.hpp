#pragma once
#include <utility>
#include <tuple>
#include "_config_node_and_vals.hpp"
#include "dispatcher/dispatcher.hpp"

namespace graph::optim {
    class OpNode : public graph::Node {
        dispatcher::OP _op;

    public:
        void set_args(
            uint64_t id, std::string name, dispatcher::OP op,
            const std::vector<Value*>& in_Values, const std::vector<Value*>& out_Values
        ) {
            this->set_node_args(id, name, in_Values, out_Values);
            _op = std::move(op);
        }
        ~OpNode() override = default;
    };

    class InputNode : public graph::Value {
    public:
        ~InputNode() override = default;
        void set_args(
            uint64_t id, const at::Tensor& tensor
        ) {
            this->set_value_args(id, tensor, nullptr, {});
        }
    };

    class OPVals : public graph::Value {
    public:
        void set_args(
            uint64_t id, const at::Tensor& tensor, Node* producer, const std::vector<Node*>& user
        ) {
            this->set_value_args(id, tensor, producer, user);
        }
        ~OPVals() override = default;
    };
}
