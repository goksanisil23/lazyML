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

// TEST(OperationTestSuite, MatMulTest_2)
// {
//     Eigen::MatrixXf matrix_1(3, 2), matrix_2(2, 3);
//     matrix_1 << 3.8, 0.05, 2.3, 3.35, 1.6, 2.6;
//     matrix_2 << 1.1, 3.5, 3.7, 0.05, 1.25, 1.;
//     needle::Tensor tensor_1(matrix_1);
//     needle::Tensor tensor_2(matrix_2);

//     needle::Tensor  res_tensor = needle::matMul(tensor_1, tensor_2);
//     Eigen::MatrixXf res_eigen{res_tensor.getNdArray()};

//     Eigen::MatrixXf expected_res(matrix_1.rows(), matrix_2.cols());
//     expected_res << 4.1825, 13.3625, 14.11, 2.6975, 12.2375, 11.86, 1.89, 8.85, 8.52;
//     EXPECT_TRUE(expected_res.isApprox(res_eigen, 0.001));
// }

// TEST(OperationTestSuite, MatMulTest_3)
// {
//     Eigen::Tensor<float, 3> tensor(2, 3, 4);

//     // Filling the tensor with values
//     int value = 1;
//     for (int i = 0; i < tensor.dimension(0); ++i)
//     {
//         for (int j = 0; j < tensor.dimension(1); ++j)
//         {
//             for (int k = 0; k < tensor.dimension(2); ++k)
//             {
//                 tensor(i, j, k) = value++;
//             }
//         }
//     }

//     // Accessing and printing the tensor elements
//     for (int i = 0; i < tensor.dimension(0); ++i)
//     {
//         for (int j = 0; j < tensor.dimension(1); ++j)
//         {
//             for (int k = 0; k < tensor.dimension(2); ++k)
//             {
//                 std::cout << "tensor(" << i << ", " << j << ", " << k << ") = " << tensor(i, j, k) << std::endl;
//             }
//         }
//     }

//     std::cout << tensor.size() << std::endl;
// }