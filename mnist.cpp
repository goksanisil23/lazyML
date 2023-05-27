#include <arpa/inet.h>
#include <fstream>
#include <iostream>
#include <vector>

#include "Eigen/Dense"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;

/**
 * @brief Read MNIST images from a file and stores them in 2D (num_samples x num_dim) Eigen matrix 
 * where num_dim is the flattened dimension (784) of the 28x28 MNIST images,
 * @param filename  MNIST training dataset filename
 * @return all images as [num_samples x (28*28)] Eigen Matrix in MNIST training dataset
 */
Eigen::MatrixXd read_mnist_images(string filename)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "Failed to open file: " << filename << endl;
        exit(1);
    }

    int magic_number, num_images, num_img_rows, num_img_cols;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = ntohl(magic_number);
    if (magic_number != 2051)
    {
        cerr << "Invalid magic number for images file: " << magic_number << endl;
        exit(1);
    }

    file.read((char *)&num_images, sizeof(num_images));
    num_images = ntohl(num_images);

    file.read((char *)&num_img_rows, sizeof(num_img_rows));
    num_img_rows = ntohl(num_img_rows);

    file.read((char *)&num_img_cols, sizeof(num_img_cols));
    num_img_cols = ntohl(num_img_cols);

    printf("num rows: %d num cols: %d\n", num_img_cols, num_img_rows);

    Eigen::MatrixXd images(num_images, num_img_rows * num_img_cols);

    for (int row = 0; row < num_images; row++)
    {
        for (int col = 0; col < num_img_rows * num_img_cols; col++)
        {
            unsigned char pixel;
            file.read((char *)&pixel, sizeof(pixel));
            images(row, col) = static_cast<double>(pixel) / 255.0;
        }
    }

    return images;
}

Eigen::VectorXi read_mnist_labels(string filename)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "Failed to open file: " << filename << endl;
        exit(1);
    }

    int magic_number, num_labels;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = ntohl(magic_number);
    if (magic_number != 2049)
    {
        cerr << "Invalid magic number for labels file: " << magic_number << endl;
        exit(1);
    }

    file.read((char *)&num_labels, sizeof(num_labels));
    num_labels = ntohl(num_labels);

    Eigen::VectorXi labels(num_labels);
    for (int i = 0; i < num_labels; i++)
    {
        unsigned char label;
        file.read((char *)&label, sizeof(label));
        // labels[i] = static_cast<int>(label);
        labels(i) = static_cast<int>(label);
    }

    return labels;
}

void viewImage(const Eigen::MatrixXd &images, const Eigen::VectorXi &labels)
{
    constexpr int num_rows = 28;
    constexpr int num_cols = 28;

    // Display the first 10 images
    cv::namedWindow("MNIST Images", cv::WINDOW_AUTOSIZE);
    for (size_t i = 0; i < images.rows(); i++)
    {
        printf("label: %d\n", labels(i));
        // Create a Mat object from the image vector
        cv::Mat image_mat(num_rows, num_cols, CV_8UC1);
        for (int r = 0; r < num_rows; r++)
        {
            for (int c = 0; c < num_cols; c++)
            {
                double pixel_value        = images(i, r * num_cols + c) * 255.0;
                image_mat.at<uchar>(r, c) = static_cast<uchar>(pixel_value);
            }
        }

        // Display the image
        cv::imshow("MNIST Images", image_mat);

        // Wait for a key press before moving on to the next image
        cv::waitKey(0);
    }
}

// h = (num_samples x num_labels)
double softmax_loss(const Eigen::MatrixXd &h, const Eigen::VectorXi &y)
{

    // h(3,y(3)) --> gives the hypothesis of GT at 3rd sample. ~ what's our belief about the GT

    double loss{0.};
    for (size_t i_sample{0}; i_sample < h.rows(); i_sample++)
    {
        double term{0.};
        for (size_t i_label{0}; i_label < h.cols(); i_label++)
        {
            term += std::exp(h(i_sample, i_label));
        }
        loss += -h(i_sample, y(i_sample)) + term;
    }

    // take the average
    return loss / y.size();
}

// Runs a single epoch = 1 pass over entire dataset
void softmax_regression_epoch(const Eigen::MatrixXd &train_images, // (num_samples x input_dim)
                              const Eigen::VectorXi &train_labels, // (num_samples)
                              Eigen::MatrixXd       &theta,        // (input_dim x num_labels)
                              const float           &learn_rate,
                              const size_t          &batch_size)
{
    // Sample a minibatch, and update the parameters with that gradient
    // So we're not actually working with the true(full) gradient
    size_t mini_batch_iters{std::floor((train_labels.size() + batch_size - 1U) / batch_size)};
    for (size_t i{0}; i < mini_batch_iters; i++)
    {
        auto x = train_images.block(
            i * batch_size, 0, (i + 1) * batch_size - 1, train_images.size());   // (batch_size x input_dim)
        auto y = train_labels.segment(i * batch_size, (i + 1) * batch_size - 1); // (batch_size)
        auto Z = (x * theta).exp();                                              // (batch_size x num_labels)

        // normalize over the labels
        Z = Z.rowwise() / Z.rowwise().sum();

        Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(batch_size, 10); // one-hot encoding
        // only the true_label index will be 1
        for (int j = 0; j < batch_size; j++)
        {
            Y(j, y(j)) = 1;
        }

        // (input_dim x batch_size) * (batch_size x num_labels) = (input_dim x num_labels)
        auto grad = x.transpose() * (Z - Y) / batch_size;

        theta -= learn_rate * grad;
    }
}

void train_softmax(const Eigen::MatrixXd &train_images,
                   const Eigen::VectorXi &train_labels,
                   const size_t          &epochs,
                   const float           &learn_rate,
                   const size_t          &batch_size)
{
    // Initial parameters of the regression
    Eigen::MatrixXd theta{Eigen::MatrixXd::Zero(28 * 28, 10)}; // (input_dim x output_dim)

    for (size_t i_epoch{0}; i_epoch < epochs; i_epoch++)
    {
        softmax_regression_epoch(train_images, train_labels, theta, learn_rate, batch_size);

        double train_loss{softmax_loss(train_images * theta, train_labels)};
    }
    // h = X ⋅ theta
    // (num_samples x input_dim) ⋅ (input_dim x num_labels)
}

int main()
{
    Eigen::MatrixXd train_images{read_mnist_images("../train-images.idx3-ubyte")};
    Eigen::VectorXi train_labels{read_mnist_labels("../train-labels.idx1-ubyte")};

    int num_images = train_images.rows();
    printf("num images: %d\n", num_images);

    // Display the first 10 images
    // viewImage(train_images, train_labels);

    constexpr size_t epochs{10};
    constexpr float  learn_rate{0.5};
    constexpr size_t batch_size{100};
    train_softmax(train_images, train_labels, epochs, learn_rate, batch_size);

    auto loss = s std::cout << "loss: " << loss << std::endl;

    return 0;
}
