#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include <unsupported/Eigen/CXX11/Tensor>

#include "Fastor/Fastor.h"

namespace needle
{

// Forward declerations
// template <size_t Dims>
class Tensor;

enum class OptBool
{
    NONE = 0,
    TRUE,
    FALSE
};

enum class ArrayApi
{
    Fastor = 0
};

constexpr bool     kLazyMode{false};
constexpr ArrayApi kArrayApi{ArrayApi::Fastor};

template <size_t... Dims>
using NdArray = Fastor::Tensor<float, Dims...>;

class TensorOp
{
  public:
    template <typename... Ts>
    auto operator()(const Ts &...tensor_inputs)
    {
        return Tensor::makeTensorFromOperation(this, tensor_inputs...);
    }

    /// Calculate forward pass of the operator. It directly executes on the raw data.
    template <typename... Ts>
    auto forward(const Ts &...array_inputs);

    /// If a node is z = x * y, right_gradient is ∂L/∂z = dz
    /// dz = starting from the end(right side) how the loss function has been effected by the output of this node (z).
    /// Since computed backwards, this dz is available
    /// @param right_gradient  ∂L/∂z = how the output of this node affects the loss
    /// @param node  Value of the node in forward iteration
    /// @return         left_gradient: result of the backward iteration: Partial (backwards) gradients to be propagated to each (forward) input node
    // std::vector<Tensor> backward(Tensor right_gradient, Tensor node);
};

/// A value computed in computation graph, i.e. output of some Operation applied to other Value objects.
class Tensor
{
  public:
    // Construct a tensor object from raw data
    template <size_t... Dims>
    Tensor(const NdArray<Dims...> ndarray_in, OptBool requires_grad = OptBool::TRUE) : cached_data(ndarray_in)
    {
        requires_grad_ = requires_grad;
    }

    template <typename... Ts>
    Tensor(TensorOp *operation, const Ts &...inputs, OptBool requires_grad = OptBool::TRUE)
    {
        operation_     = operation;
        requires_grad_ = requires_grad;

        if (requires_grad_ == OptBool::NONE)
        {
            for (auto input : inputs)
            {
                if (input.requires_grad_ == OptBool::TRUE)
                {
                    requires_grad_ = OptBool::TRUE;
                    break;
                }
            }
        }
    }

    template <typename... Ts>
    static Tensor makeTensorFromOperation(TensorOp *operation, const Ts &...inputs)
    {
        Tensor tensor(operation, inputs...);
        if (!kLazyMode)
        {
            tensor.realizeCachedData(inputs...);
        }
        return tensor;
    }

    template <typename... Ts>
    void realizeCachedData(const Ts &...inputs)
    // NdArray<Dims> realizeCachedData()
    {
        // If cached data is not None (already been computed)
        if (cached_data_.size() != 0)
        {
            return cached_data_;
        }
        else
        {
            // std::vector<NdArray> realized_inputs;
            for (const auto input : inputs)
            {
                std::cout << input.size() << std::endl;
                // realized_inputs.push_back(input->realizeCachedData());
            }
            // cached_data_ = operation_->forward(realized_inputs);
            // return cached_data_;
        }
    }

    // Create a new tensor that shares the data but detaches from the graph
    // Tensor<Dims> detach()
    // {
    //     return makeConst(this->realizeCachedData());
    // }

    // static Tensor makeConst(const NdArray &data, OptBool requires_grad = OptBool::FALSE)
    // {
    //     Tensor<Dims> tensor;
    //     tensor.init(nullptr, {}, 1, data, requires_grad);
    //     return tensor;
    // }

    // bool isLeaf() const
    // {
    //     return (operation_ == nullptr);
    // }

    // NdArray<Dims> getNdArray()
    // {
    //     NdArray<Dims> data = this->realizeCachedData();
    //     if (kArrayApi == ArrayApi::Eigen)
    //     {
    //         return data;
    //     }
    //     else
    //     {
    //         throw("we should only have Eigen for now");
    //     }
    // }

  public:
    TensorOp           *operation_{nullptr};
    std::vector<Tensor> inputs_; // TODO: avoid this copy
    size_t              num_outputs_{1};

    // Fields for dynamic computation
    // NdArray<Dims...> cached_data_;
    OptBool requires_grad_{};
};

} // namespace needle