#include "autograd.h"

namespace needle
{
Tensor Operation::operator()(const std::vector<Value *> &inputs)
{
    throw std::runtime_error("Not implemented");
}

Eigen::MatrixXd Operation::forward(const std::vector<Eigen::MatrixXd> &inputs)
{
    throw std::runtime_error("Not implemented");
}

std::vector<Value> Operation::backward(Value right_gradient, Value node)
{
    throw std::runtime_error("Not implemented");
}

Tensor TensorOperation::operator()(const std::vector<Value *> &inputs)
{
    return Tensor::makeFromOperation(this, inputs);
}

/// A value computed in computation graph, i.e. output of some Operation applied to other Value objects.

void Value::init(Operation                  *operation,
                 const std::vector<Value *> &inputs,
                 const size_t                num_outputs,
                 const Eigen::MatrixXd      &cached_data,
                 const OptBool               requires_grad_opt)
{
    operation_ = operation;
    inputs_    = inputs;

    if (requires_grad_opt == OptBool::NONE)
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
Eigen::MatrixXd Value::realizeCachedData()
{
    if (cached_data_.size() != 0)
    {
        return cached_data_;
    }
    else
    {
        std::vector<Eigen::MatrixXd> realized_inputs;
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

Tensor::Tensor(const Eigen::MatrixXd &cached_data, const OptBool requires_grad)
{
    init(nullptr, {}, 1, cached_data, requires_grad);
}

Tensor Tensor::makeFromOperation(Operation *operation, const std::vector<Value *> &inputs)
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

Tensor Tensor::makeConst(const Eigen::MatrixXd &data, OptBool requires_grad)
{
    Tensor tensor;
    tensor.init(nullptr, {}, 1, data, requires_grad);
    return tensor;
}

Eigen::MatrixXd Tensor::getEigen()
{
    auto data = this->realizeCachedData();
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