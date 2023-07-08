#include "autograd.h"

#include <cassert>

namespace needle
{
// Operator to element-wise divide 2 tensors.
class ElWiseDiv : public TensorOp
{
  public:
    template <size_t... Dims>
    NdArray<Dims...> forward(const NdArray<Dims...> in1, const NdArray<Dims...> in2)
    {
        return in1 / in2;
    }
};

template <size_t... Dims>
Tensor<Dims...> elwiseDivide(Tensor<Dims...> &a, Tensor<Dims...> &b)
{
    return ElWiseDiv()(a, b);
}

// // Operator to divide elements of tensor with a scalar.
// class ScalarDiv : public TensorOperation
// {
//   public:
//     explicit ScalarDiv(const float scalar) : scalar_{scalar}
//     {
//     }

//     NdArray forward(const std::vector<NdArray> &inputs) override
//     {
//         assert(inputs.size() == 1);

//         return inputs[0] / scalar_;
//     }

//   private:
//     float scalar_;
// };

// Tensor scalarDivide(Tensor &a, const float scalar)
// {
//     return ScalarDiv(scalar)(std::vector<Value *>{&a});
// }

// // Operator to matrix-multiply 2 tensors, currently only supports 3 dimensions.
// class MatMul : public TensorOperation
// {
//   public:
//     NdArray forward(const std::vector<NdArray> &inputs) override
//     {
//         assert(inputs.size() == 2);
//         assert((inputs[0].dimension(0) == 1) || (inputs[1].dimension(0) == 1) ||
//                (inputs[0].dimension(0) == inputs[1].dimension(0)));

//         // e.g. if tensors are (2,3,4) & (2,4,5), we take 4 of first, 4 of second tensor, resulting in (2,3,5)
//         //  since they contraction indices need to match per matrix multiplication
//         if (inputs[0].dimension(0) == inputs[1].dimension(0))
//         {
//             NdArray res(inputs[0].dimension(0), inputs[0].dimension(1), inputs[1].dimension(2));
//             constexpr Eigen::array<Eigen::IndexPair<int>, 1> contraction_pair = {Eigen::IndexPair<int>(1, 0)};
//             for (int i = 0; i < inputs[0].dimension(0); i++)
//             {
//                 auto m1        = inputs[0].chip(i, 0);
//                 auto m2        = inputs[1].chip(i, 0);
//                 auto m1m2      = m1.contract(m2, contraction_pair);
//                 res.chip(i, 0) = m1m2;
//             }
//             return res;
//         }
//         else
//         {
//             int max_dim_input = (inputs[0].dimension(0) > inputs[1].dimension(0)) ? 0 : 1;

//             NdArray res(inputs[max_dim_input].dimension(0), inputs[0].dimension(1), inputs[1].dimension(2));
//             constexpr Eigen::array<Eigen::IndexPair<int>, 1> contraction_pair = {Eigen::IndexPair<int>(1, 0)};
//             for (int i = 0; i < inputs[max_dim_input].dimension(0); i++)
//             {
//                 auto m1        = inputs[0].chip((0 == max_dim_input) ? i : 0, 0);
//                 auto m2        = inputs[1].chip((1 == max_dim_input) ? i : 0, 0);
//                 auto m1m2      = m1.contract(m2, contraction_pair);
//                 res.chip(i, 0) = m1m2;
//             }
//             return res;
//         }
//     }
// };

// Tensor matMul(Tensor &a, Tensor &b)
// {
//     return MatMul()(std::vector<Value *>{&a, &b});
// }

// //  Sum of array elements over given axes
// class Summation : public TensorOperation
// {
//   public:
//     explicit Summation(const int axes) : axes_{axes}
//     {
//     }

//     NdArray forward(const std::vector<NdArray> &inputs) override
//     {
//         assert(inputs.size() == 1);

//         if (axes_ == -1)
//         {
//             NdArray                 res(1, 1, 1);
//             Eigen::Tensor<float, 0> sum = inputs[0].sum();
//             res(0, 0, 0)                = sum(0);
//             return res;
//         }
//         else
//         {
//             // Eigen::Tensor<float, 1> sum = inputs[0].sum(Eigen::array<int, 2>({0, axes_}));
//             // NdArray                 res(1, 1, sum.dimension(0));
//             // res.chip(0, 0) = sum;

//             Eigen::Tensor<float, 1> sum = inputs[0].sum(Eigen::array<int, 2>({0, axes_}));
//             NdArray                 res(1, 1, sum.dimension(0));
//             for (int i = 0; i < sum.dimension(0); i++)
//             {
//                 res(0, 0, i) = sum(i);
//             }
//             return res;
//         }
//     }

//   private:
//     int axes_{-1}; // -1 :reduction among all axes
// };

// Tensor summation(Tensor &a, const int axes = -1)
// {
//     return Summation(axes)(std::vector<Value *>{&a});
// }

} // namespace needle