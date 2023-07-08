# Manual Neural Networks

- Initially we used linear hypothesis: `h_θ(x) = θ*x`
- this maps `N`-dim inputs to `K`-dim outputs (classes)
	- by forming `K` linear functions of the input
	- whichever gives the largest result is chosen to be class prediction
- But what if there is no linear classifier that could classify our data?
	- We introduce "feature mapping" to our hypothesis:
	- `h_θ(x)` = `θ * φ(x)` --> `φ` is mapping from `R^N` to `R^D`
		- `φ` is feature mapping (mapping input from input space to some higher/lower dim)
	- `h_θ` becomes a linear function of these features
	- selection of this feature mapping can be done with NNs or manual feature engineering
- To exceed the linear hypothesis performance, we introduce some non-linear function to feature vector:
	- `φ(x) = σ(W*x)`, `W=(n x d)`, `σ = R^n -> R^d` (some non-linear function)
	- this is immediately more powerful than linear classifier:
		- e.g. choose `W` randomly, and let `σ`  be cosine func --> this works great for many problems
- So now our hypothesis is: `h_θ(x) = θ * σ(W*x)`
	- We have parameters `θ,W` to be optimized here
	- And this is what NNs will do essentially: they'll co-optimize these 2 parameter sets (both linear classifier and feature vector parameters) 
		- THis is actually the simplest NN: 2 layer NN 
- Why adding non-linearity is so powerful?
	- Universal function approximator: NN function is piecewise linear (consider Relu)
	- By choosing cut-off point and slope of Relu and adding enough of these, we can approximate different regions of any smooth f(x) function that we're approximating.

<img src="https://raw.githubusercontent.com/goksanisil23/lazyML/main/ManualNeuralNetworks/resources/universal_func_approx.png" width=30% height=50%>


## Implementation
- We have analytically derived the gradient of the loss function. We use this gradient within **stochastic gradient descent**. The term stochastic comes from the fact that we divide the entire training dataset into mini-batches, compute the gradient on the minibatch and update the model parameters per each batch, instead of doing a single update by using the entire dataset which can have large memory requirements. 
By randomly sampling mini-batches, SGD approximates the true gradient of the loss function by using a noisy estimate.
The noise in SGD allows escaping local minima and faster convergence, but shows more fluctuation compared to regular gradient descent.
- Our hypothesis `h_θ(x)` can take any real value. For multi-class classification, we would prefer to have a probability where the sum of all output nodes adds to 1.0. This is enabled by the **softmax** function:

<img src="https://raw.githubusercontent.com/goksanisil23/lazyML/main/ManualNeuralNetworks/resources/softmax.png" width=13% height=10%>

Then, we define the loss function to be the negative log probability of our network predicting the true class label. (also called *cross-entropy loss*)

<img src="https://raw.githubusercontent.com/goksanisil23/lazyML/main/ManualNeuralNetworks/resources/cross_entropy_loss.png" width=60% height=50%>

where `h_y` is the hypothesis value corresponding to output node giving y as classification.
## Build
- Extract the MNIST dataset to the path of this binary.
```sh
clang++ mnist.cpp -o mnist -I/usr/include/eigen3 -I/usr/include/SDL2/ -lSDL2 -O3 -Wall -std=c++17
```
