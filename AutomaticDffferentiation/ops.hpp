#include "autograd.h"

#include <cassert>

namespace needle
{
// Operator to element-wise divide 2 tensors.
class ElWiseDiv : public TensorOperation
{
  public:
    NdArray forward(const std::vector<NdArray> &inputs) override
    {
        assert(inputs.size() == 2);

        return inputs[0] / inputs[1];
    }
};

Tensor elwiseDivide(Tensor &a, Tensor &b)
{
    return ElWiseDiv()(std::vector<Value *>{&a, &b});
}

// Operator to divide elements of tensor with a scalar.
class ScalarDiv : public TensorOperation
{
  public:
    explicit ScalarDiv(const float scalar) : scalar_{scalar}
    {
    }

    NdArray forward(const std::vector<NdArray> &inputs) override
    {
        assert(inputs.size() == 1);

        return inputs[0] / scalar_;
    }

  private:
    float scalar_;
};

Tensor scalarDivide(Tensor &a, const float scalar)
{
    return ScalarDiv(scalar)(std::vector<Value *>{&a});
}

// Operator to matrix-multiply 2 tensors
class MatMul : public TensorOperation
{
  public:
    NdArray forward(const std::vector<NdArray> &inputs) override
    {
        assert(inputs.size() == 2);
        assert((inputs[0].dimension(0) == 1) || (inputs[1].dimension(0) == 1) ||
               (inputs[0].dimension(0) == inputs[1].dimension(0)));

        // e.g. if tensors are (2,3,4) & (2,4,5), we take 4 of first, 4 of second tensor, resulting in (2,3,5)
        //  since they contraction indices need to match per matrix multiplication
        if (inputs[0].dimension(0) == inputs[1].dimension(0))
        {
            Eigen::Tensor<float, 3> res(inputs[0].dimension(0), inputs[0].dimension(1), inputs[1].dimension(2));
            constexpr Eigen::array<Eigen::IndexPair<int>, 1> contraction_pair = {Eigen::IndexPair<int>(1, 0)};
            for (int i = 0; i < inputs[0].dimension(0); i++)
            {
                auto m1        = inputs[0].chip(i, 0);
                auto m2        = inputs[1].chip(i, 0);
                auto m1m2      = m1.contract(m2, contraction_pair);
                res.chip(i, 0) = m1m2;
            }
            return res;
        }
        else
        {
            int max_dim_input = (inputs[0].dimension(0) > inputs[1].dimension(0)) ? 0 : 1;

            Eigen::Tensor<float, 3> res(
                inputs[max_dim_input].dimension(0), inputs[0].dimension(1), inputs[1].dimension(2));
            constexpr Eigen::array<Eigen::IndexPair<int>, 1> contraction_pair = {Eigen::IndexPair<int>(1, 0)};
            for (int i = 0; i < inputs[max_dim_input].dimension(0); i++)
            {
                auto m1        = inputs[0].chip((0 == max_dim_input) ? i : 0, 0);
                auto m2        = inputs[1].chip((1 == max_dim_input) ? i : 0, 0);
                auto m1m2      = m1.contract(m2, contraction_pair);
                res.chip(i, 0) = m1m2;
            }
            return res;
        }
    }
};

Tensor matMul(Tensor &a, Tensor &b)
{
    return MatMul()(std::vector<Value *>{&a, &b});
}

} // namespace needle