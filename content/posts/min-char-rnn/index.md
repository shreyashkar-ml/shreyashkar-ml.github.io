---
title: "Decoding Karpathy's min-char-rnn (character level Recurrent Neural Network)"
date: 2024-09-22
draft: false
math: true
toc: true
---

Recurrent Neural Networks (RNN) have existed for long at this point, and RNNs without attention mechanism (plain-simple RNN architecture) are no longer the hottest thing either.

Still, RNN represents one of the first step towards understanding training for sequential data input, where the context of previous inputs are crucial for predicting the next output.

Karpathy's introduction to the [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) along with the attached [Character Level RNN](https://gist.github.com/karpathy/d4dee566867f8291f086) are among the best resources to get started with RNN. However, as I progressed my way through the implementation of min-char-rnn, I realized that while the blogpost suffices for an intuitive understanding of RNN, and the code works through the implementation from scratch, a lot of heavy-lifting in terms of manual backpropagation implementation, and flow of gradient during training to update the weights and parameters of the models are left for the readers to understand on their own.

We'll be going through the steps one-by-one, mainly focusing on backpropagation calculation in detail to understand the behind-the-hood working of RNN.

## **Overview of RNN**
Recurrent Neural Networks are at the core an attempt to develop an internal structure that is appropriate for a particular task domain using internal 'hidden' units which are not part of the input or output vectors.

Learning becomes more interesting but more difficult when we introduce hidden units whose actual desired states are not specified by the task. The simplest for of the learning procedure is for layered networks which have a layer of inputs at the bottom; any number of intermediate layers; and a layer of output units at the top.

An input vector is presented to the network by setting the states of the input units.

![RNN Unrolled](rnn-unrolled.png "RNN Unrolled")

Then the stats of the units in each layer are determined by applying steps as followed for input vector $ x_t $ : 
- **Hidden State Calculation:**
</br> $ h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h) $

- **Output and Softmax:**
<br> $ y_t = W_{hy} \cdot h_t + b_y $
<br> $ p_t = \frac{ \exp(y_t) }{ \sum exp(y_t) } $ </br>

where, $ h_{t-1} $ represents the hidden state input from previous states, $ b_h $ represents the biased term in hidden state calculation, and $ p_t $ represents the softmax output from the output vector $ y_t $.

![RNN Illustration](rnn_illustration.jpeg "RNN Illustration")
*An illustration of character level RNN from Andrej Karpathy's blogpost*

RNNs are particularly effective in tasks like language modeling, machine translation, etc. where the context of previous characters are crucial for predicting the next one.

Now, let's work through min-char-rnn code one-step at a time:
### `lossFun` Function:
This function runs both forward and backward passes through the RNN and computes the loss and gradients.

```python
def lossFun(inputs, targets, hprev):
  """
  Runs forward and backward passes through the RNN.

  inputs, targets: Lists of integers. For some i, inputs[i] is the input
                   character (encoded as an index to the ix_to_char map) and
                   targets[i] is the corresponding next character in the
                   training data (similarly encoded).
  hprev: Hx1 array of initial hidden state.
  returns: loss, gradients on model parameters, and last hidden state.
  """
```
**Inputs**:
- `inputs`: Indices representing the input characters.
- `targets`: Indices representing the next characters in the sequence
- `hprev`: Initial hidden state from the previous sequence

**Outputs**:
- `loss`: Cross-entropy loss
- Gradients for the weights and biases (`dWxh, dWhh, dWhy, dbh, dby`)
- The last hidden state (`hs[len(inputs)-1]`)

### Forward Pass

The forward pass computes the hidden states and the outputs at each time step.

```python
# Initialize storage for variables needed for forward and backward passes
xs, hs, ys, ps = {}, {}, {}, {}
hs[-1] = np.copy(hprev) # Initialize with the given hidden state
loss = 0
```
For each time step $ t $:
1. **Input Encoding**: The input characters are converted into one-hot encoding vectors for input into the model.
```python
xs[t] = np.zeros((vocab_size, 1)) # one-hot encoding
xs[t][inputs[t]] = 1
```
2. **Hidden State Calculation**: The hidden states to capture the task information and develop/emulate an internal structure is computed using:
$$ h_t = \tanh(W_{xh}\cdot x_t + W_{hh} \cdot h_{t-1} + b_h) $$
```python
hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
```
3. **Output and Softmax**: We first calculate the unnormalized scores (`ys[t]`) and then converet those into softmax probabilities (`ps[t]`) for output:
$$ y_t = W_{hy} \cdot h_t + b_y $$
$$ p_t = \frac{\exp(y_t)}{\sum \exp(y_t)} $$
```python
ys[t] = np.dot(Why, hs[t]) + by
ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
```
4. **Loss Calculation**: The cross-entropy loss at each time step is then added up using:
```python
loss += -np.log(ps[t][targets[t],0])
```
### Backpropagation through time (BPTT)

1. **Gradient of Loss w.r.t Softmax Output (`ps[t]`):**
For each time step t:
The loss at time step $ t $ is given by:
$$ Loss_t = - \log(p_{t,target}) $$
The derivative of the loss w.r.t softmax probabilities $p_t$ is:
$$ \frac{\partial Loss_t}{\partial p_t} = p_t - 1_{target} $$
Where:
- $ p_t $ is the softmax probability vector at time step t.
- $ 1_{target} $ is a one-hot vector with a 1 at the index of the target character.
```python
dy = np.copy(ps[t])
dy[targets[t]] -= 1
```

2. **Gradient w.r.t Output Weights (`Why`) and Bias (`by`):**
The output $ y_t $ at each time step is computed as:
$$ y_t = W_{hy} \cdot h_t + b_y $$
The gradients of the loss w.r.t the weights and biases are given by:
$$ \frac{ \partial Loss_t }{\partial W_{hy}} =
\sum_t \left(\frac{ \partial Loss_t }{ \partial y_t } \cdot \frac{ \partial y_t }{ \partial W_{hy}} \right)= \sum_t( p_t - 1_{target}).h_t^T $$
$$ \frac {\partial Loss_t }{\partial b_y} = \sum_t(p_t - 1_{target}) $$
```python
dWhy += np.dot(dy, hs[t].T)
dby += dy
```

3. **Gradient w.r.t Hidden State (`h_t`):**
To backpropagate into the hidden state, we need to account for both the current time step's gradient and the incoming gradient from the next time step:
$$ \frac{ \partial Loss_t }{\partial h_t} = W_{hy}^T \cdot \frac{\partial Loss_t}{ \partial y_t} + \frac{\partial Loss_{t+1}}{\partial h_t} $$
Where:
- $ \frac{\partial Loss_{t+1}}{\partial h_t} $ is the gradient passed back from the next time step.
```python
dh = np.dot(Why.T, dy) + dhnext
```

4. **Gradient w.r.t Activation Function (`tanh`):**
The hidden state is computed using the $ tanh $ activation function:
$$ h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h) $$
The input to the `tanh` activation function is:
$$ a_t = W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h $$
The gradient through the $ tanh $ function is:
$$ dhraw = \frac{\partial Loss_t}{\partial a_t} = \frac{ \partial Loss_t }{\partial ({W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h })} = (1 - h_{t}^2) \odot \frac{ \partial Loss_t }{\partial h_t} $$
Here, $ (1-h_t^2) $ is the derivative of $ tanh(h_t) $.
```python
dhraw = (1- hs[t]*hs[t]) * dh
```


5. **Gradient w.r.t Input Weights (`Wxh`), Hidden Weights (`Whh`), and Hidden Bias (`bh`):**
Now, compute the gradients w.r.t weights and biases connecting the inputs and hidden states:
The input to the `tanh` activation function is:
$$ a_t = W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h $$
- For input-to-hidden weights $ W_{xh} $:
$$ \frac{\partial Loss_t}{\partial W_{xh}} = \sum_t \left(\frac{\partial Loss_t}{\partial a_t} \cdot \frac{\partial a_t}{\partial W_{xh}} \right)= \sum_t \left(\text{dhraw} \cdot x_t^T \right) $$
- For hidden-to-hidden weights $ W_{hh} $:
$$ \frac{\partial Loss_t}{\partial W_{hh}} = \sum_t \left(\frac{ \partial Loss_t}{\partial a_t} \cdot \frac{ \partial a_t }{\partial W_{hh}} \right) = \sum_t (\text{dhraw} \cdot h_{t-1}^T) $$
- For hidden bias $ b_h $:
$$
\frac{\partial Loss_t}{\partial b_h} = \sum_t \frac{\partial Loss}{\partial a_t} = \sum_tdhraw
$$
```python
dbh += dhraw
dWxh += np.dot(dhraw, xs[t].T)
dWhh += np.dot(dhraw, hs[t-1].T)
```

6. **Gradient Propagation to Previous Time Step (`dhnext`):**
Propogate the gradient back to the previous time step:
$$
\frac{ \partial Loss_t}{\partial h_{t-1}} = W_{hh}^T \cdot \frac{\partial Loss_t}{\partial a_t} = W_{hh}^T \cdot dhraw
$$
```python
dhnext = np.dot(Whh.T, dhraw)
```

#### Summary:
- **Step 1:** Compute gradient of the loss w.r.t the output probabilities.
- **Step 2:** Calculate the gradients for the weights and biases connecting hidden states to outputs.
- **Step 3:** Propagate the gradient through the hidden state, taking into account the contribution from the next time step.
- **Step 4:** Backpropagate through the $ tanh $ activation function.
- **Step 5:** Compute the gradients w.r.t the weights and biases connecting the inputs and hidden states, as well as the hidden-to-hidden weights.
- **Step 6:** Propagate the gradient back to the previous time step's hidden state.

These steps iteratively update the gradient accumulations by iterating backward through the time steps of the sequence.

Now, with all the heavy-lifting stuffs taken care of, I think understanding RNNs in-depth would be a lot easier than before. Here's the [min-char-rnn code](https://github.com/shreyashkar-ml/Pytorch-learning/blob/main/min_char_rnn_explained.ipynb) commented and explained a bit more in details to work through the code without losing the touch with explanation.

## References

1. [Learning representations by backpropagating errors](https://gwern.net/doc/ai/nn/1986-rumelhart-2.pdf)
2. [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
3. [Character Level RNN](https://gist.github.com/karpathy/d4dee566867f8291f086)
4. [Tutorial on Recurrent Neural Networks](https://dennybritz.com/posts/wildml/recurrent-neural-networks-tutorial-part-1/)
5. [A Nice in-depth walkthrough of Character RNN](https://eli.thegreenplace.net/2018/understanding-how-to-implement-a-character-based-rnn-language-model/)
6. [RNN Illustrations](https://kvitajakub.github.io/2016/04/14/rnn-diagrams/)