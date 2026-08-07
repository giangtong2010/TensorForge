#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include "tensor.hpp"

namespace graph {
    class Value;

    class Node {
        uint64_t _node_id;
        std::string _node_name;

        std::vector<Value*> _in_values;
        std::vector<Value*> _out_values;

    protected:
        virtual ~Node() = default;
        void set_node_args(
                uint64_t id, std::string name, 
                const std::vector<Value*>& in_Values, const std::vector<Value*>& out_Values
            );
        Node() = default;
    };

    class Value {
        uint64_t _id;
        Node* _producer;
        std::vector<Node*> _user;

        cpp20::Dtype _dtype;
        cpp20::Device _device;

        std::vector<int64_t> _size;
        std::vector<int64_t> _stride;

    protected:
        virtual ~Value() = default;
        void set_value_args(
            uint64_t id, const at::Tensor& tensor, Node* producer, const std::vector<Node*>& user
        );

        Value() = default;
    };
}
