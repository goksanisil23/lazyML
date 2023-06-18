# Automatic Differentiation
A general ML problem can be divided into 3 core elements:
- Hypothesis class, loss function, optimization method
Computing the gradient of the loss function w.r.t hypothesis class parameters is the most common operation in ML.

Numeric differentiation is powerful, but it requires 2 forward passes to compute the gradient `f(θ+ε)-f(θ-ε)/2ε`, hence inefficient.

**Forward mode auto diff**: We build a computation graph and recycle computed notes (chain rule allows this) as we traverse along the graph.

<img src="https://raw.githubusercontent.com/goksanisil23/lazyML/main/AutomaticDffferentiation/resources/forward_mode_AD.png" width=30% height=50%>

The downside of forward mode AD is that for `f : R^n -> R^k`, we need `n` forward AD passes to get the gradient w.r.t each input.   

**Reverse mode auto diff**: 
1- Start with forward pass, evaluate intermediate variables, store values and dependencies of intermediate variables.
2- Compute partial derivatives of the output, w.r.t intermediate variables = Adjoints

<img src="https://raw.githubusercontent.com/goksanisil23/lazyML/main/AutomaticDffferentiation/resources/reverse_mode_AD.png" width=30% height=50%>



## TODO: Render the computation graph


## RUN
g++ -o test_needle tests/test_needle.cpp -lgtest -lgtest_main -pthread