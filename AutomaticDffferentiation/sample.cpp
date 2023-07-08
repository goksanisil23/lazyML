#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

namespace needle
{
class NDArray
{
  public:
    // NDArray implementation goes here
};

class Device
{
  public:
    // Device implementation goes here
};

class CPUDevice : public Device
{
  public:
    std::string repr()
    {
        return "needle.cpu()";
    }

    // size_t hash()
    // {
    //     return std::hash<std::string>()(repr());
    // }

    bool operator==(const CPUDevice &other)
    {
        return true;
    }

    bool enabled()
    {
        return true;
    }
};

CPUDevice cpu_device()
{
    return CPUDevice();
}

std::vector<Device> all_devices()
{
    return {cpu_device()};
}

class Op
{
  public:
    virtual ~Op()
    {
    }
    virtual NDArray              compute(const std::vector<NDArray> &args)              = 0;
    virtual std::vector<NDArray> gradient(const NDArray &out_grad, const NDArray &node) = 0;
};

class TensorOp : public Op
{
  public:
    NDArray compute(const std::vector<NDArray> &args) override
    {
        // Perform computation and return the result
        return NDArray();
    }

    std::vector<NDArray> gradient(const NDArray &out_grad, const NDArray &node) override
    {
        // Compute the gradients and return them
        return std::vector<NDArray>();
    }
};

class Value
{
  private:
    std::shared_ptr<Op> op;
    std::vector<Value>  inputs;
    NDArray             cached_data;
    bool                requires_grad;

  public:
    Value(std::shared_ptr<Op> op, std::vector<Value> inputs, bool requires_grad)
        : op(op), inputs(inputs), requires_grad(requires_grad)
    {
    }

    NDArray realize_cached_data()
    {
        // Check if the cached data is already computed
        if (!cached_data.is_empty())
        {
            return cached_data;
        }

        // Compute the cached data by recursively realizing inputs' data
        std::vector<NDArray> input_data;
        for (const auto &input : inputs)
        {
            input_data.push_back(input.realize_cached_data());
        }

        // Compute and cache the data
        cached_data = op->compute(input_data);
        return cached_data;
    }

    bool is_leaf()
    {
        return op == nullptr;
    }

    void backward(const NDArray &out_grad = NDArray())
    {
        // Compute gradients of variables
        compute_gradient_of_variables(*this, out_grad);
    }

    // Additional methods and properties can be added here as needed
};

class Tensor : public Value
{
  public:
    Tensor(std::shared_ptr<Op> op, std::vector<Value> inputs, bool requires_grad) : Value(op, inputs, requires_grad)
    {
    }

    // Additional methods and properties specific to Tensor can be added here
};

void compute_gradient_of_variables(Value &output_tensor, const NDArray &out_grad)
{
    // Map from node to a list of gradient contributions from each output node
    std::unordered_map<Value, std::vector<NDArray>> grad_map;
    grad_map[output_tensor] = {out_grad};

    // Traverse the computation graph in reverse order and compute gradients
    std::vector<Value> stack = {output_tensor};
    while (!stack.empty())
    {
        Value node = stack.back();
        stack.pop_back();

        // Compute gradients for this node
        const std::vector<NDArray> &grad_contributions = grad_map[node];
        std::vector<NDArray>        grads = node.op->gradient(grad_contributions[0], node.realize_cached_data());

        // Assign gradients to inputs
        for (size_t i = 0; i < node.inputs.size(); ++i)
        {

            Value input = node.inputs[i];
            if (grad_map.find(input) == grad_map.end())
            {
                grad_map[input] = {grads[i]};
            }
            else
            {
                grad_map[input].push_back(grads[i]);
            }

            // If the input is not a leaf, add it to the stack for further processing
            if (!input.is_leaf())
            {
                stack.push_back(input);
            }
        }
    }
}
} // namespace needle

int main()
{
    // Create a tensor operation
    std::shared_ptr<needle::Op> tensor_op = std::make_shared<needle::TensorOp>();
    std::vector<needle::Value>  inputs;
    bool                        requires_grad = true;
    needle::Tensor              tensor(tensor_op, inputs, requires_grad);

    // Perform computations and backward pass
    tensor.realize_cached_data();
    tensor.backward();

    return 0;
}
