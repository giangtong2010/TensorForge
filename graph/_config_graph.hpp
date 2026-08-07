#pragma once
#include "_config_node_and_vals.hpp"

namespace graph::digraph {
    class Digraph {
        uint64_t _graph_id;
        uint64_t _node_id = 0;

        std::vector<std::unique_ptr<Node>> _nodes = {};
        std::vector<std::unique_ptr<Value>> _Values = {};

    protected:
        virtual void add_node(Node* node) = 0;
        virtual void add_value(Value* value) = 0;
        virtual void create_graph(uint64_t graph_id) = 0;
        virtual ~Digraph() = default;

        Digraph() = default;
        uint64_t& get_node_id() {return _node_id;}
        std::vector<std::unique_ptr<Node>> get_nodes() {return _nodes;}
        std::vector<std::unique_ptr<Value>> get_values() {return _Values;}
    };
}
