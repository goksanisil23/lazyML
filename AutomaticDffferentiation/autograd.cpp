#include "autograd.h"

namespace needle
{

// --------------- Operation --------------- //
Tensor Operation::operator()(const std::vector<Value *> &inputs)
{
    throw std::runtime_error("Not implemented");
}

NdArray Operation::forward(const std::vector<NdArray> &inputs)
{
    throw std::runtime_error("Not implemented");
}

std::vector<Value> Operation::backward(Value right_gradient, Value node)
{
    throw std::runtime_error("Not implemented");
}

// --------------- TensorOperation --------------- //
Tensor TensorOperation::operator()(const std::vector<Value *> &inputs)
{
    return Tensor::makeTensorFromOperation(this, inputs);
}

// --------------- Value --------------- //
/// A value computed in computation graph, i.e. output of some Operation applied to other Value objects.
void Value::init(Operation                  *operation,
                 const std::vector<Value *> &inputs,
                 const size_t                num_outputs,
                 const NdArray              &cached_data,
                 const OptBool               requires_grad_opt)
{
    operation_     = operation;
    inputs_        = inputs;
    requires_grad_ = requires_grad_opt;

    if (requires_grad_ == OptBool::NONE)
    {
        for (auto input : inputs)
        {
            if (input->requires_grad_ == OptBool::TRUE)
            {
                requires_grad_ = OptBool::TRUE;
                break;
            }
        }
    }
    operation_   = operation;
    num_outputs_ = num_outputs;
    cached_data_ = cached_data;
}

// Run compute to realize cached data, avoid recomputation if it's already computed before
NdArray Value::realizeCachedData()
{
    if (cached_data_.size() != 0)
    {
        return cached_data_;
    }
    else
    {
        std::vector<NdArray> realized_inputs;
        for (const auto input : inputs_)
        {
            realized_inputs.push_back(input->realizeCachedData());
        }
        cached_data_ = operation_->forward(realized_inputs);
        return cached_data_;
    }
}

bool Value::isLeaf() const
{
    return (operation_ == nullptr);
}

// --------------- Tensor --------------- //
Tensor::Tensor(const NdArray &cached_data, const OptBool requires_grad)
{
    init(nullptr, {}, 1, cached_data, requires_grad);
}

Tensor Tensor::makeTensorFromOperation(Operation *operation, const std::vector<Value *> &inputs)
{
    Tensor tensor;
    tensor.init(operation, inputs);
    if (!kLazyMode)
    {
        tensor.realizeCachedData();
    }
    return tensor;
}

// Create a new tensor that shares the data but detaches from the graph
Tensor Tensor::detach()
{
    return Tensor::makeConst(this->realizeCachedData());
}

Tensor Tensor::makeConst(const NdArray &data, OptBool requires_grad)
{
    Tensor tensor;
    tensor.init(nullptr, {}, 1, data, requires_grad);
    return tensor;
}

NdArray Tensor::getNdArray()
{
    NdArray data = this->realizeCachedData();
    if (kArrayApi == ArrayApi::Eigen)
    {
        return data;
    }
    else
    {
        throw("we should only have Eigen for now");
    }
}
} // namespace needle