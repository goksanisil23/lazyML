#include <gtest/gtest.h>

#include "../autograd.h"
#include "../ops.hpp"

#include <vector>

TEST(VectorDivisionTest, DivideVectors)
{
    Eigen::Matrix3d matrix_1, matrix_2;
    matrix_1 << 2, 4, 6, 8, 10, 12, 14, 16, 18;
    matrix_2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    needle::Tensor tensor_1(matrix_1);
    needle::Tensor tensor_2(matrix_2);

    needle::Tensor  res_tensor = needle::divide(tensor_1, tensor_2);
    Eigen::MatrixXd res_eigen{res_tensor.getEigen()};

    Eigen::Matrix3d expected_res;
    expected_res << 2, 2, 2, 2, 2, 2, 2, 2, 2;

    EXPECT_TRUE(expected_res.isApprox(res_eigen, 0.001));
}