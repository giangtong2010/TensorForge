#pragma once
#include "optim_nodes_and_vals.hpp"
#include "_config_graph.hpp"

namespace graph::optimization_graph {
    class optimization_graph : public graph::digraph::Digraph {

        void add_node(Node* node) override;
        void add_value(Value* value) override;

    public:
        optimization_graph() = default;
        ~optimization_graph() override = default;
        
        void add_input_node(const at::Tensor& tensor);
        void add_op_node(dispatcher::OP op);
        void create_graph(uint64_t graph_id) override;
    };
}
