#include "autograd.h"

#include <cassert>

namespace needle
{
// Operator to element-wise divide 2 tensors.
class ElWiseDiv : public TensorOperation
{
  public:
    Eigen::MatrixXd forward(const std::vector<Eigen::MatrixXd> &inputs) override
    {
        assert(inputs.size() == 2);
        assert(inputs[0].size() == inputs[1].size());

        return inputs[0].array() / inputs[1].array();
    }
};

Tensor divide(Tensor &a, Tensor &b)
{
    return ElWiseDiv()(std::vector<Value *>{&a, &b});
}
} // namespace needle