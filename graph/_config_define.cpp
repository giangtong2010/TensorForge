#include "_config_graph.hpp"
#include "_config_node_and_vals.hpp"

namespace graph {
    void Node::set_node_args(
        uint64_t id, std::string name,
        const std::vector<Value*>& in_values, const std::vector<Value*>& out_values
    ) {
        _node_id = id;
        _node_name = name;
        _in_values = in_values;
        _out_values = out_values;
    }
    void Value::set_value_args(
        uint64_t id, const at::Tensor& tensor, Node* producer, const std::vector<Node*>& user
    ) {
        _producer = producer;
        _user = user;
        _id = id;

        _dtype = tensor.get_dtype();
        _device = tensor.get_device();

        _size = tensor.get_size();
        _stride = tensor.get_stride();
    }
}
