#include <arpa/inet.h>
#include <fstream>
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "render.hpp"

constexpr size_t kBatchSize{100};
constexpr float  kBatchSizeF{static_cast<float>(kBatchSize)};
constexpr float  kLearnRate{0.5};
constexpr size_t kNumEpochs{10};
constexpr size_t kImgWidth{28};
constexpr size_t kImgHeight{28};
constexpr size_t kInputDim{kImgWidth * kImgHeight}; // MNIST input data dimensions (28x28 pixels)
constexpr size_t kHiddenDim{500};
constexpr size_t kNumLabels{10};

struct LossAndError
{
    double loss{0.F};
    double error{0.F};
};

void showResults(const Eigen::MatrixXd &test_images, const Eigen::MatrixXd &W1, const Eigen::MatrixXd &W2);

/**
 * @brief Read MNIST images from a file and stores them in 2D (num_samples x num_dim) Eigen matrix 
 * where num_dim is the flattened dimension (784) of the 28x28 MNIST images,
 * @param filename  MNIST training dataset filename
 * @return all images as [num_samples x (28*28)] Eigen Matrix in MNIST training dataset
 */
Eigen::MatrixXd read_mnist_images(std::string filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    int magic_number, num_images, num_img_rows, num_img_cols;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = ntohl(magic_number);
    if (magic_number != 2051)
    {
        std::cerr << "Invalid magic number for images file: " << magic_number << std::endl;
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
        // std::cout << "loading image " << row << std::endl;
        for (int col = 0; col < num_img_rows * num_img_cols; col++)
        {
            unsigned char pixel;
            file.read((char *)&pixel, sizeof(pixel));
            images(row, col) = static_cast<double>(pixel) / 255.0;
        }
    }

    return images;
}

// Reads the labels from MNIST dataset into 1D Eigen vector
Eigen::VectorXi read_mnist_labels(std::string filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    int magic_number, num_labels;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = ntohl(magic_number);
    if (magic_number != 2049)
    {
        std::cerr << "Invalid magic number for labels file: " << magic_number << std::endl;
        exit(1);
    }

    file.read((char *)&num_labels, sizeof(num_labels));
    num_labels = ntohl(num_labels);

    Eigen::VectorXi labels(num_labels);
    for (int i = 0; i < num_labels; i++)
    {
        // std::cout << "loading label " << i << std::endl;
        unsigned char label;
        file.read((char *)&label, sizeof(label));
        labels(i) = static_cast<int>(label);
    }

    return labels;
}

// h = (num_samples x num_labels)
// y = (num_samples x 1)
// Calculates the softmax loss and error terms,
// given the hypothesis matrix(h) and the ground truth labels of the dataset (y)
LossAndError softmaxLossAndError(const Eigen::MatrixXd &h, const Eigen::VectorXi &y)
{

    // h(3,y(3)) --> gives the hypothesis of true label at 3rd sample. ~ what's our belief about the GT

    double loss{0.};
    double error{0.};
    for (size_t i_sample{0}; i_sample < h.rows(); i_sample++)
    {
        double term{0.};
        // term is cumulative belief regarding all labels, with our hypothesis
        for (size_t i_label{0}; i_label < h.cols(); i_label++)
        {
            term += std::exp(h(i_sample, i_label));
        }
        loss += -h(i_sample, y(i_sample)) + std::log(term);

        // Error
        // In our hypothesis, check the label with highest confidence and compare with GT.
        // If not the same as GT, increment the error by 1
        size_t best_hypo_class;
        h.row(i_sample).maxCoeff(&best_hypo_class);
        error += static_cast<double>(y(i_sample) != best_hypo_class);
    }

    // take the average
    return {loss / y.size(), error / y.size()};
}

// Runs a single epoch = 1 pass over entire dataset
void softmaxRegressionEpoch(const Eigen::MatrixXd &train_images, // (num_samples x input_dim)
                            const Eigen::VectorXi &train_labels, // (num_samples)
                            Eigen::MatrixXd       &theta)              // (input_dim x num_labels)

{
    // Sample a minibatch, and update the parameters with that gradient
    // So we're not actually working with the true(full) gradient
    static size_t mini_batch_iters{static_cast<size_t>(
        std::floor((static_cast<float>(train_labels.size() + kBatchSize - 1U)) / static_cast<float>(kBatchSize)))};

    for (size_t i{0}; i < mini_batch_iters; i++)
    {
        Eigen::MatrixXd x = train_images.block<kBatchSize, kInputDim>(i * kBatchSize, 0); // (batch_size x input_dim)
        Eigen::VectorXi y = train_labels.segment(i * kBatchSize, kBatchSize);             // (batch_size)
        Eigen::MatrixXd Z = (x * theta).array().exp();                                    // (batch_size x num_labels)

        // normalize over the labels
        for (size_t row_idx{0}; row_idx < Z.rows(); row_idx++)
        {
            Z.row(row_idx) = Z.row(row_idx).array() / Z.row(row_idx).sum();
        }

        Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(kBatchSize, 10); // one-hot encoding (batch_size, num_labels)
        // only the true_label index will be 1
        for (size_t j = 0U; j < kBatchSize; j++)
        {
            Y(j, y(j)) = 1.0;
        }

        // (input_dim x batch_size) * (batch_size x num_labels) = (input_dim x num_labels)
        auto grad = (x.transpose() * (Z - Y));

        theta -= (kLearnRate / kBatchSizeF) * grad;
    }
}

void trainSoftmax(const Eigen::MatrixXd &train_images,
                  const Eigen::VectorXi &train_labels,
                  const Eigen::MatrixXd &test_images,
                  const Eigen::VectorXi &test_labels)
{
    // Initial parameters of the regression
    Eigen::MatrixXd theta{Eigen::MatrixXd::Zero(kInputDim, 10)}; // (input_dim x output_dim)

    for (size_t i_epoch{0}; i_epoch < kNumEpochs; i_epoch++)
    {
        softmaxRegressionEpoch(train_images, train_labels, theta);

        // h = X ⋅ theta
        // (num_samples x input_dim) ⋅ (input_dim x num_labels)
        LossAndError le_train{softmaxLossAndError(train_images * theta, train_labels)};
        LossAndError le_test{softmaxLossAndError(test_images * theta, test_labels)};
        std::cout << "epoch " << i_epoch << " tr_loss: " << le_train.loss << " tr_error: " << le_train.error
                  << " te_loss: " << le_test.loss << " te_error: " << le_test.error << std::endl;
    }
}

void nnEpoch(const Eigen::MatrixXd &train_images, // (num_samples x input_dim)
             const Eigen::VectorXi &train_labels, // (num_samples)
             Eigen::MatrixXd       &W1,           // (input_dim x num_hidden)
             Eigen::MatrixXd       &W2)                 // (num_hidden x num_labels)
{
    static size_t mini_batch_iters{static_cast<size_t>(
        std::floor((static_cast<float>(train_labels.size() + kBatchSize - 1U)) / static_cast<float>(kBatchSize)))};

    for (size_t i{0}; i < mini_batch_iters; i++)
    {
        Eigen::MatrixXd x = train_images.block<kBatchSize, kInputDim>(i * kBatchSize, 0); // (batch_size x input_dim)
        Eigen::VectorXi y = train_labels.segment(i * kBatchSize, kBatchSize);             // (batch_size)

        // Z1 = ReLu(X*W1)
        Eigen::MatrixXd Z1 = (x * W1);                       // (batch_size x num_hidden)
        Z1                 = (Z1.array() < 0).select(0, Z1); // set entries smaller than 0 to 0 since Relu

        // G2 = normalize(exp(Z1*W2)) - Y
        // Normalize over labels
        Eigen::MatrixXd G2 = (Z1 * W2).array().exp(); // (batch_size x num_labels)
        for (size_t row_idx{0}; row_idx < G2.rows(); row_idx++)
        {
            G2.row(row_idx) = G2.row(row_idx).array() / G2.row(row_idx).sum();
        }
        // Y = one-hot encoding (batch_size, num_labels)
        Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(kBatchSize, 10);
        // only the true_label index will be 1
        for (size_t j = 0U; j < kBatchSize; j++)
        {
            Y(j, y(j)) = 1.0;
        }
        G2 -= Y;

        // G1 = 1{Z1>0}∘(G2*W2^T)
        // ∘ -> element wise mul
        // 1{Z1>0} -> binary matrix based on Z1 elements being >0
        Eigen::MatrixXd G1 =
            ((Z1.array() > 0).cast<double>()).array() * (G2 * W2.transpose()).array(); // (batch_size x num_hidden)

        Eigen::MatrixXd grad_1 = x.transpose() * G1 / kBatchSizeF;
        Eigen::MatrixXd grad_2 = Z1.transpose() * G2 / kBatchSizeF;
        W1 -= kLearnRate * grad_1;
        W2 -= kLearnRate * grad_2;
    }
}

// Training 2-layer neural network, where the hidden layer is ReLu
void trainNN(const Eigen::MatrixXd &train_images,
             const Eigen::VectorXi &train_labels,
             const Eigen::MatrixXd &test_images,
             const Eigen::VectorXi &test_labels)
{
    // Initialize weights with random numbers
    Eigen::MatrixXd W1 = Eigen::MatrixXd::Random(kInputDim, kHiddenDim);
    Eigen::MatrixXd W2 = Eigen::MatrixXd::Random(kHiddenDim, kNumLabels);

    for (size_t i_epoch{0}; i_epoch < kNumEpochs; i_epoch++)
    {
        nnEpoch(train_images, train_labels, W1, W2);

        // h = Relu(X ⋅ W1)⋅W2
        // (num_samples x input_dim) ⋅ (input_dim x hidden_dim) ⋅ (hidden_dim x num_label)
        auto            f_train{train_images * W1};
        Eigen::MatrixXd feature_vector_train = (f_train.array() < 0).select(0, f_train); // (num_samples x hidden_dim)
        LossAndError    le_train{softmaxLossAndError(feature_vector_train * W2, train_labels)};

        auto            f_test{test_images * W1};
        Eigen::MatrixXd feature_vector_test = (f_test.array() < 0).select(0, f_test); // (num_samples x hidden_dim)
        LossAndError    le_test{softmaxLossAndError(feature_vector_test * W2, test_labels)};

        std::cout << "epoch " << i_epoch << " tr_loss: " << le_train.loss << " tr_error: " << le_train.error
                  << " te_loss: " << le_test.loss << " te_error: " << le_test.error << std::endl;
    }

    // Show some results with visualization
    showResults(test_images, W1, W2);
}

void viewImage(const Eigen::VectorXd &img)
{
    unsigned char image_buffer[kInputDim * 4]; // RGBA format
    for (size_t r = 0; r < kImgHeight; r++)
    {
        for (size_t c = 0; c < kImgWidth; c++)
        {
            size_t idx{(r * kImgWidth + c)};
            double pixel_value = img(idx) * 255.0;
            idx *= 4;
            image_buffer[idx]     = 255;
            image_buffer[idx + 1] = static_cast<unsigned char>(pixel_value);
            image_buffer[idx + 2] = static_cast<unsigned char>(pixel_value);
            image_buffer[idx + 3] = static_cast<unsigned char>(pixel_value);
        }
    }

    render(image_buffer);
}

void showResults(const Eigen::MatrixXd &test_images, const Eigen::MatrixXd &W1, const Eigen::MatrixXd &W2)
{
    for (size_t i = 0; i < test_images.rows(); ++i)
    {
        Eigen::MatrixXd img = test_images.row(i);

        // Predict the label of this image with the weights
        // Using 2 layer network we trained with
        auto            f{img * W1};
        Eigen::MatrixXd feature_vector = (f.array() < 0).select(0, f); // (num_samples x hidden_dim)
        Eigen::MatrixXd h              = feature_vector * W2;
        size_t          best_hypo_class;
        h.row(0).maxCoeff(&best_hypo_class);
        std::cout << "predicted class: " << best_hypo_class << std::endl;
        viewImage(img.row(0));
    }
}

int main()
{
    Eigen::MatrixXd train_images{read_mnist_images("./train-images.idx3-ubyte")};
    Eigen::VectorXi train_labels{read_mnist_labels("./train-labels.idx1-ubyte")};
    Eigen::MatrixXd test_images{read_mnist_images("./t10k-images.idx3-ubyte")};
    Eigen::VectorXi test_labels{read_mnist_labels("./t10k-labels.idx1-ubyte")};

    int num_images = train_images.rows();
    printf("num images: %d\n", num_images);

    trainSoftmax(train_images, train_labels, test_images, test_labels);
    trainNN(train_images, train_labels, test_images, test_labels);

    return 0;
}
