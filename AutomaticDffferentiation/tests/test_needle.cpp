#include <gtest/gtest.h>

#include "../autograd.h"
#include "../ops.hpp"

#include <vector>

namespace
{
template <typename T>
using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar, int rank, typename sizeType>
auto eigenTensorToMatrix(const Eigen::Tensor<Scalar, rank> &tensor, const sizeType rows, const sizeType cols)
{
    return Eigen::Map<const MatrixType<Scalar>>(tensor.data(), rows, cols);
}
} // namespace

TEST(OperationTestSuite, ElwiseDivisionTest)
{
    needle::NdArray matrix_1(1, 2, 3);
    matrix_1.setValues({{{3.3, 4.35, 1.2}, {2.45, 0.95, 2.55}}});
    needle::NdArray matrix_2(1, 2, 3);
    matrix_2.setValues({{{4.6, 4.35, 4.8}, {0.65, 0.7, 4.4}}});

    needle::Tensor tensor_1(matrix_1);
    needle::Tensor tensor_2(matrix_2);

    needle::Tensor res_tensor = needle::elwiseDivide(tensor_1, tensor_2);

    needle::NdArray         res_ndarray{res_tensor.getNdArray()};
    Eigen::Tensor<float, 2> res_eigen_tensor = res_ndarray.chip(0, 0);
    Eigen::MatrixXf         res_eigen_mtx    = eigenTensorToMatrix(res_eigen_tensor, 2, 3);

    Eigen::MatrixXf expected_res(2, 3);
    expected_res << 0.717391304348, 1., 0.25, 3.769230769231, 1.357142857143, 0.579545454545;

    EXPECT_TRUE(expected_res.isApprox(res_eigen_mtx));
}

TEST(OperationTestSuite, ScalarDivisionTest)
{
    needle::NdArray matrix(1, 1, 2);
    matrix.setValues({{{1.7, 1.45}}});
    needle::Tensor tensor(matrix);

    needle::Tensor res_tensor = needle::scalarDivide(tensor, 12.F);

    needle::NdArray         res_ndarray{res_tensor.getNdArray()};
    Eigen::Tensor<float, 2> res_eigen_tensor = res_ndarray.chip(0, 0);
    Eigen::MatrixXf         res_eigen_mtx    = eigenTensorToMatrix(res_eigen_tensor, 1, 2);

    Eigen::MatrixXf expected_res(1, 2);
    expected_res << 0.141666666667, 0.120833333333;
    EXPECT_TRUE(expected_res.isApprox(res_eigen_mtx, 0.001));
}

TEST(OperationTestSuite, MatMulTest_1)
{
    needle::NdArray matrix_1(1, 3, 3);
    needle::NdArray matrix_2(1, 3, 3);
    matrix_1.setValues({{{4.95, 1.75, 0.25}, {4.15, 4.25, 0.3}, {0.3, 0.4, 2.1}}});
    matrix_2.setValues({{{1.35, 2.2, 1.55}, {3.85, 4.8, 2.6}, {1.15, 0.85, 4.15}}});
    needle::Tensor tensor_1(matrix_1);
    needle::Tensor tensor_2(matrix_2);

    needle::Tensor res_tensor = needle::matMul(tensor_1, tensor_2);

    needle::NdArray         res_ndarray{res_tensor.getNdArray()};
    Eigen::Tensor<float, 2> res_eigen_tensor = res_ndarray.chip(0, 0);
    Eigen::MatrixXf         res_eigen_mtx    = eigenTensorToMatrix(res_eigen_tensor, 3, 3);

    Eigen::MatrixXf expected_res(3, 3);
    expected_res << 13.7075, 19.5025, 13.26, 22.31, 29.785, 18.7275, 4.36, 4.365, 10.22;
    EXPECT_TRUE(expected_res.isApprox(res_eigen_mtx, 0.001));
}

TEST(OperationTestSuite, MatMulTest_2)
{
    needle::NdArray matrix_1(1, 3, 2), matrix_2(1, 2, 3);
    matrix_1.setValues({{{3.8, 0.05}, {2.3, 3.35}, {1.6, 2.6}}});
    matrix_2.setValues({{{1.1, 3.5, 3.7}, {0.05, 1.25, 1.}}});
    needle::Tensor tensor_1(matrix_1);
    needle::Tensor tensor_2(matrix_2);

    needle::Tensor res_tensor = needle::matMul(tensor_1, tensor_2);

    needle::NdArray         res_ndarray{res_tensor.getNdArray()};
    Eigen::Tensor<float, 2> res_eigen_tensor = res_ndarray.chip(0, 0);
    Eigen::MatrixXf         res_eigen_mtx    = eigenTensorToMatrix(res_eigen_tensor, 3, 3);

    Eigen::MatrixXf expected_res(3, 3);
    expected_res << 4.1825, 13.3625, 14.11, 2.6975, 12.2375, 11.86, 1.89, 8.85, 8.52;
    EXPECT_TRUE(expected_res.isApprox(res_eigen_mtx, 0.001));
}

// Matrix multiplication of 3-d arrays of same channel length
TEST(OperationTestSuite, MatMulTest_3)
{
    needle::NdArray matrix_1(6, 3, 2);
    matrix_1.setValues({{{4., 2.15}, {1.25, 1.35}, {0.75, 1.6}},
                        {{2.9, 2.15}, {3.3, 4.1}, {2.5, 0.25}},
                        {{2.9, 4.35}, {1.2, 3.5}, {3.55, 3.95}},
                        {{2.55, 4.35}, {4.25, 0.2}, {3.95, 3.4}},
                        {{2.2, 2.05}, {0.95, 1.8}, {2.7, 2.}},
                        {{0.45, 1.1}, {3.15, 0.7}, {2.9, 1.95}}});
    needle::NdArray matrix_2(6, 2, 3);
    matrix_2.setValues({{{2.7, 4.05, 0.1}, {1.75, 3.05, 2.3}},
                        {{0.55, 4.1, 2.3}, {4.45, 2.35, 2.55}},
                        {{1.2, 3.95, 4.6}, {4.2, 3.5, 3.35}},
                        {{2.55, 4.4, 2.05}, {2.4, 0.6, 4.65}},
                        {{2.95, 0.8, 0.6}, {0.45, 1.3, 0.75}},
                        {{1.25, 2.1, 0.4}, {0.85, 3.5, 3.7}}});
    needle::Tensor tensor_1(matrix_1);
    needle::Tensor tensor_2(matrix_2);

    needle::Tensor res_tensor = needle::matMul(tensor_1, tensor_2);

    needle::NdArray expected_res(6, 3, 3);
    expected_res.setValues({{{14.5625, 22.7575, 5.345}, {5.7375, 9.18, 3.23}, {4.825, 7.9175, 3.755}},
                            {{11.1625, 16.9425, 12.1525}, {20.06, 23.165, 18.045}, {2.4875, 10.8375, 6.3875}},
                            {{21.75, 26.68, 27.9125}, {16.14, 16.99, 17.245}, {20.85, 27.8475, 29.5625}},
                            {{16.9425, 13.83, 25.455}, {11.3175, 18.82, 9.6425}, {18.2325, 19.42, 23.9075}},
                            {{7.4125, 4.425, 2.8575}, {3.6125, 3.1, 1.92}, {8.865, 4.76, 3.12}},
                            {{1.4975, 4.795, 4.25}, {4.5325, 9.065, 3.85}, {5.2825, 12.915, 8.375}}});

    needle::NdArray res_ndarray{res_tensor.getNdArray()};
    for (int i = 0; i < matrix_1.dimension(0); i++)
    {
        Eigen::Tensor<float, 2> res_eigen_tensor     = res_ndarray.chip(i, 0);
        Eigen::MatrixXf         res_eigen_mtx        = eigenTensorToMatrix(res_eigen_tensor, 3, 3);
        Eigen::Tensor<float, 2> exp_res_eigen_tensor = expected_res.chip(i, 0);
        Eigen::MatrixXf         exp_res_eigen_mtx    = eigenTensorToMatrix(exp_res_eigen_tensor, 3, 3);
        EXPECT_TRUE(exp_res_eigen_mtx.isApprox(res_eigen_mtx, 0.001));
    }
}

// Matrix multiplication of 3-d arrays of different channel length (1st matrix single, 2nd matrix multi channel)
TEST(OperationTestSuite, MatMulTest_4)
{
    needle::NdArray matrix_1(1, 3, 2);
    matrix_1.setValues({{{1.9, 1.9}, {4.8, 4.9}, {3.25, 3.75}}});
    needle::NdArray matrix_2(3, 2, 3);
    matrix_2.setValues({{{1.25, 1.8, 1.95}, {3.75, 2.85, 2.25}},
                        {{1.75, 2.7, 3.3}, {2.95, 1.55, 3.85}},
                        {{4.2, 3.05, 3.35}, {3.3, 4.75, 2.1}}});
    needle::Tensor tensor_1(matrix_1);
    needle::Tensor tensor_2(matrix_2);

    needle::Tensor res_tensor = needle::matMul(tensor_1, tensor_2);

    needle::NdArray expected_res(3, 3, 3);
    expected_res.setValues({{{9.5, 8.835, 7.98}, {24.375, 22.605, 20.385}, {18.125, 16.5375, 14.775}},
                            {{8.93, 8.075, 13.585}, {22.855, 20.555, 34.705}, {16.75, 14.5875, 25.1625}},
                            {{14.25, 14.82, 10.355}, {36.33, 37.915, 26.37}, {26.025, 27.725, 18.7625}}});

    needle::NdArray res_ndarray{res_tensor.getNdArray()};
    for (int i = 0; i < matrix_2.dimension(0); i++)
    {
        Eigen::Tensor<float, 2> res_eigen_tensor     = res_ndarray.chip(i, 0);
        Eigen::MatrixXf         res_eigen_mtx        = eigenTensorToMatrix(res_eigen_tensor, 3, 3);
        Eigen::Tensor<float, 2> exp_res_eigen_tensor = expected_res.chip(i, 0);
        Eigen::MatrixXf         exp_res_eigen_mtx    = eigenTensorToMatrix(exp_res_eigen_tensor, 3, 3);
        EXPECT_TRUE(exp_res_eigen_mtx.isApprox(res_eigen_mtx, 0.001));
    }
}

// Matrix multiplication of 3-d arrays of different channel length (2nd matrix single, 1st matrix multi channel)
TEST(OperationTestSuite, MatMulTest_5)
{
    needle::NdArray matrix_1(3, 3, 2);
    matrix_1.setValues({{{3.4, 2.95}, {0.25, 1.95}, {4.4, 4.4}},
                        {{0.55, 1.1}, {0.75, 1.55}, {4.1, 1.2}},
                        {{1.5, 4.05}, {1.5, 1.55}, {2.3, 1.25}}});
    needle::NdArray matrix_2(1, 2, 3);
    matrix_2.setValues({{{2.2, 0.65, 2.5}, {2.5, 1.3, 0.15}}});
    needle::Tensor tensor_1(matrix_1);
    needle::Tensor tensor_2(matrix_2);

    needle::Tensor res_tensor = needle::matMul(tensor_1, tensor_2);

    needle::NdArray expected_res(3, 3, 3);
    expected_res.setValues({{{14.855, 6.045, 8.9425}, {5.425, 2.6975, 0.9175}, {20.68, 8.58, 11.66}},
                            {{3.96, 1.7875, 1.54}, {5.525, 2.5025, 2.1075}, {12.02, 4.225, 10.43}},
                            {{13.425, 6.24, 4.3575}, {7.175, 2.99, 3.9825}, {8.185, 3.12, 5.9375}}});

    needle::NdArray res_ndarray{res_tensor.getNdArray()};
    for (int i = 0; i < matrix_1.dimension(0); i++)
    {
        Eigen::Tensor<float, 2> res_eigen_tensor     = res_ndarray.chip(i, 0);
        Eigen::MatrixXf         res_eigen_mtx        = eigenTensorToMatrix(res_eigen_tensor, 3, 3);
        Eigen::Tensor<float, 2> exp_res_eigen_tensor = expected_res.chip(i, 0);
        Eigen::MatrixXf         exp_res_eigen_mtx    = eigenTensorToMatrix(exp_res_eigen_tensor, 3, 3);
        EXPECT_TRUE(exp_res_eigen_mtx.isApprox(res_eigen_mtx, 0.001));
    }
}