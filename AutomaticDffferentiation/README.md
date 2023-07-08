# Automatic Differentiation
A general ML problem can be divided into 3 core elements:
- Hypothesis class, loss function, optimization method
Computing the gradient of the loss function w.r.t hypothesis class parameters is the most common operation in ML.

Numeric differentiation is powerful, but it requires 2 forward passes to compute the gradient `f(θ+ε)-f(θ-ε)/2ε`, hence inefficient.

**Forward mode auto diff**: We build a computation graph and recycle computed notes (chain rule allows this) as we traverse along the graph.

<img src="https://raw.githubusercontent.com/goksanisil23/lazyML/main/AutomaticDffferentiation/resources/forward_mode_AD.png" width=30% height=50%>

The downside of forward mode AD is that for `f : R^n -> R^k`, we need `n` forward AD passes to get the gradient w.r.t each input.   

**Reverse mode auto diff**: 
<<<<<<< HEAD

<img src="https://raw.githubusercontent.com/goksanisil23/lazyML/main/AutomaticDffferentiation/resources/reverse_mode_AD.png" width=40% height=50%>
=======
<img src="https://raw.githubusercontent.com/goksanisil23/lazyML/main/AutomaticDffferentiation/resources/reverse_mode_AD.png" width=30% height=50%>
>>>>>>> 838e3782e315da62ccaa51dd3525891e6d16f21f

There are 2 main components to consider while doing backprop, due to chain rule: adjoints & local node gradients
 - `backward_gradient = adjoint * local_gradient` 
 - adjoints capture what has happened on the right side of the graph
 - local node gradients capture whats currently happening in the node.

If a node is `z = f(x,y) = x * y`, right_gradient is `∂L/∂z = dz`. --> This is called adjoint: Gradient of the loss function, w.r.t output of a specific layer.
dz = starting from the end (right side) how the loss function has been effected by the output of this node (z). Since computed backwards, this dz is available

<img src="https://raw.githubusercontent.com/goksanisil23/lazyML/main/AutomaticDffferentiation/resources/single_node_backprop.png" width=30% height=50%>

Given a node `v_i`, a partial adjoint is defined as `∂L/∂v_i`. If node `v_i` is connected to multiple nodes towards right, partial adjoint of `v_i` is computed by sum of the partial derivatives of those nodes w.r.t `v_i` (look at `v_2` below:)

<img src="https://raw.githubusercontent.com/goksanisil23/lazyML/main/AutomaticDffferentiation/resources/computation_graph.png" width=30% height=50%>
<img src="https://raw.githubusercontent.com/goksanisil23/lazyML/main/AutomaticDffferentiation/resources/adjoint.png" width=30% height=50%>

How we compute backward gradient:
- assume a node is `f(x,y)`. We want to compute `∂L/∂x` and `∂L/∂y` (backwards gradient per each input to this node)
- `∂L/∂x = (∂L/∂f(x,y)) * (∂f(x,y)/∂x)`
    - We know the `∂L/∂f(x,y)` since via back-prop, it's available.
    - We just need to compute `∂f/∂x`, which is straightforward since we know exactly what `f(x,y)` is which is the operation in this node.

When dealing with back-prop, topological sort allows us to have `∂L/∂f(x,y)` available, because sorting allows us to start from the end of the graph and run backwards towards the dependencies(inputs) of the nodes.

Once the backward gradient is calculated, backward_grad*adjoint is distributed to the input of the current node, and the process is continued towards left.
-> result we're passing now becomes *one of* the **partial adjoint** on the next node on the left.

The core of backprop can be summarized as below:
<img src="https://raw.githubusercontent.com/goksanisil23/lazyML/main/AutomaticDffferentiation/resources/backprop_core.png" width=30% height=50%>


Given a node `v_i`, a partial adjoint is defined as `∂L/∂v_i`. If node `v_i` is connected to multiple nodes towards right, partial adjoint of `v_i` is computed by sum of the partial derivatives of those nodes w.r.t `v_i` (look at `v_2` below:)

<img src="https://raw.githubusercontent.com/goksanisil23/lazyML/main/AutomaticDffferentiation/resources/computation_graph.png" width=30% height=50%>
<img src="https://raw.githubusercontent.com/goksanisil23/lazyML/main/AutomaticDffferentiation/resources/adjoint.png" width=30% height=50%>

How we compute backward gradient:
- assume a node is `f(x,y)`. We want to compute `∂L/∂x` and `∂L/∂y` (backwards gradient per each input to this node)
- `∂L/∂x = (∂L/∂f(x,y)) * (∂f(x,y)/∂x)`
    - We know the `∂L/∂f(x,y)` since via back-prop, it's available.
    - We just need to compute `∂f/∂x`, which is straightforward since we know exactly what `f(x,y)` is which is the operation in this node.

When dealing with back-prop, topological sort allows us to have `∂L/∂f(x,y)` available, because sorting allows us to start from the end of the graph and run backwards towards the dependencies(inputs) of the nodes.

Once the backward gradient is calculated, backward_grad*adjoint is distributed to the input of the current node, and the process is continued towards left.
-> result we're passing now becomes *one of* the **partial adjoint** on the next node on the left.

The core of backprop can be summarized as below:
<img src="https://raw.githubusercontent.com/goksanisil23/lazyML/main/AutomaticDffferentiation/resources/backprop_core.png" width=100% height=50%>


In summary, an epoch of SGD of 2 layer NN with auto-diff back-propagation looks like:
```python
    for i in range(num_batches):
        batch_indices = [i * batch : (i+1) * batch]
        x = Tensor(X[batch_indices, :]) # input data
        Z = relu(x.matmul(W1)).matmul(W2) # our hypothesis
        yy = y[batch_indices] # labels
        y_one_hot = np.zeros((batch, y.max() + 1))
        y_one_hot[np.arange(batch), yy] = 1
        y_one_hot = Tensor(y_one_hot)
        loss = softmax_loss(Z, y_one_hot) # loss is a Tensor
        loss.backward() # gradient of all the operations are computed
        W1 = Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
        W2 = Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())
```
Weights are simply updated with the partial derivatives of the nodes (operations) they're acting on.