#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace needle
{

// Forward declerations
class Value;
class Tensor;

enum class OptBool
{
    NONE = 0,
    TRUE,
    FALSE
};

enum class ArrayApi
{
    Eigen = 0
};

constexpr bool     kLazyMode{false};
constexpr ArrayApi kArrayApi{ArrayApi::Eigen};
using NdArray = Eigen::Tensor<float, 3>; // TODO: We only support 3d-arrays now

class Operation
{
  public:
    virtual Tensor operator()(const std::vector<Value *> &inputs);

    /// Calculate forward pass of the operator. It directly executes on the raw data.
    virtual NdArray forward(const std::vector<NdArray> &inputs);

    /// If a node is z = x * y, right_gradient is ∂L/∂z = dz
    /// dz = starting from the end(right side) how the loss function has been effected by the output of this node (z).
    /// Since computed backwards, this dz is available
    /// @param right_gradient  ∂L/∂z = how the output of this node affects the loss
    /// @param node  Value of the node in forward iteration
    /// @return         left_gradient: result of the backward iteration: Partial (backwards) gradients to be propagated to each (forward) input node
    virtual std::vector<Value> backward(Value right_gradient, Value node);
};

class TensorOperation : public Operation
{
  public:
    Tensor operator()(const std::vector<Value *> &inputs) override;
};

/// A value computed in computation graph, i.e. output of some Operation applied to other Value objects.
class Value
{
  public:
    void init(Operation                  *operation,
              const std::vector<Value *> &inputs,
              const size_t                num_outputs       = 1,
              const NdArray              &cached_data       = NdArray(),
              const OptBool               requires_grad_opt = OptBool::NONE);

    // static Value makeFromOperation(Operation *operation, const std::vector<Value *> &inputs)
    // {
    //     Value value;
    //     value.init(operation, inputs);

    //     if (!kLazyMode)
    //     {
    //         if (value.requires_grad_ != OptBool::TRUE)
    //         {
    //             return value.detach(); // TODO: WTF is this?
    //         }
    //         value.realizeCachedData();
    //     }
    //     return value;
    // }

    // Run compute to realize cached data, avoid recomputation if it's already computed before
    NdArray realizeCachedData();

    bool isLeaf() const;

  public:
    Operation           *operation_{nullptr};
    std::vector<Value *> inputs_; // TODO: avoid this copy
    size_t               num_outputs_{1};
    // Fields for dynamic computation
    NdArray cached_data_;
    OptBool requires_grad_{};
};

class Tensor : public Value
{
  public:
    Tensor(const NdArray &cached_data = NdArray(), const OptBool requires_grad = OptBool::TRUE);

    static Tensor makeTensorFromOperation(Operation *operation, const std::vector<Value *> &inputs);

    // Create a new tensor that shares the data but detaches from the graph
    Tensor detach();

    static Tensor makeConst(const NdArray &data, OptBool requires_grad = OptBool::FALSE);

    NdArray getNdArray();
};
} // namespace needle