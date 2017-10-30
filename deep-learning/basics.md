# Normalizing inputs

* Zero out or subtract out the mean of your every data: $$X-=\mu$$
* Each data divided by sample variance: $$X/=\sigma^2$$
* Use the same $$\mu$$ and $$\sigma^2$$ values to normalize your testing data as well. you don't need to estimate the new values for both of them. Just make sure your training and testing data go through the same transformation.


[1] [Andrew Ng's Coursera video ](https://www.coursera.org/learn/deep-neural-network/lecture/lXv6U/normalizing-inputs)

# Why normalizing inputs?

Make all your features on the same scale, hopefully will make your loss function or cost function easier to optimize, especially when using SGD to optimize. See the picture.
![](/assets/Screen Shot 2017-10-28 at 10.14.07 PM.png)

[1] [Andrew Ng's Coursera video ](https://www.coursera.org/learn/deep-neural-network/lecture/lXv6U/normalizing-inputs)


# Vanishing and exploding gradients

One of the problems of training neural network, especially very deep neural networksis data vanishing and exploding gradients. What that means is that when you're training a very deep network your derivatives or your slopes can sometimes get either very, very big or very, very small, maybe even exponentially small, and this makes training difficult.

* if weights matrix $$\mathit W^{[i]} \gt \mathit{I}$$ (identity matrix), the output of activation function will be increased exponentially as a function of layers.
* if weights matrix $$\mathit W^{[i]} \lt \mathit{I}$$ (identity matrix), the output of activation function will be decreased exponentially as a function of layers.

# How to solve vanishing and exploding gradients problems?

A partial solution would be a better weight initialization scheme.

One reasonable thing to do would be to set the variance of $$W_i$$ to be equal to $$\frac{1}{n}$$, where $$n$$ is the number of input features that's going into a neuron. 
For example, if the activation is __RELU__, then we can do this in practice. 

$$
W^{[l]} = np.random.randn(shape) * np.sqrt(\frac{2}{n^{[l-1]}})
$$

This would cause output of neurons also take on a similar scale and this doesn't solve, but it definitely helps reduce the vanishing, exploding gradients problem because it's trying to set each of the weight matrices $$W$$ you know so that it's not too much bigger than 1 and not too much less than 1 so it doesn't explode or vanish too quickly.

If the activation is __tanh__, we will use $$\sqrt\frac{1}{n^{[l-1]}}$$ as the variance.
Others may use Xavier initialization as $$\sqrt \frac{1}{n^{[l-1]}+n^{[n]}}$$

# What will happen if initialize all weights in the network to zero?

In general, initializing all the weights to zero results in the network failing to break symmetry. This means that every neuron in each layer will learn the same thing, and you might as well be training a neural network with  n[l]=1
  for every layer, and the network is no more powerful than a linear classifier such as logistic regression.
  
 
**What you should remember**:
- The weights $$W^{[l]}$$ should be initialized randomly to break symmetry. 
- It is however okay to initialize the biases $$b^{[l]}$$ to zeros. Symmetry is still broken so long as $$W^{[l]}$$ is initialized randomly.
- Initializing weights to very large random values does not work well, it will generate very high cost at the beginning.
This is because with large random-valued weights, the last activation (sigmoid) outputs results that are very close to 0 or 1 for some examples, and when it gets that example wrong it incurs a very high loss for that example. Indeed, when  log(a[3])=log(0) , the loss goes to infinity.
Hopefully intializing with small random  values does better. The important question is: how small should be these random values be? Lets find out in the next part!

# L2 regularization on deep learning?

**What you should remember** -- the implications of L2-regularization on:
- The cost computation:
    - A regularization term is added to the cost
- The backpropagation function:
    - There are extra terms in the gradients with respect to weight matrices
- Weights end up smaller ("weight decay"): 
    - Weights are pushed to smaller values.

# What's dropout?
Dropout is a widely used regularization technique that is specific to deep learning. It randomly shuts down some neurons in each iteration.The idea behind drop-out is that at each iteration, you train a different model that uses only a subset of your neurons. With dropout, your neurons thus become less sensitive to the activation of one other specific neuron, because that other neuron might be shut down at any time.


**What you should remember about dropout:**
- Dropout is a regularization technique.
- You only use dropout during training. Don't use dropout (randomly eliminate nodes) during test time.
- Apply dropout both during forward and backward propagation.
- During training time, divide each dropout layer by keep_prob to keep the same expected value for the activations. For example, if keep_prob is 0.5, then we will on average shut down half the nodes, so the output will be scaled by 0.5 since only the remaining half are contributing to the solution. Dividing by 0.5 is equivalent to multiplying by 2. Hence, the output now has the same expected value. You can check that this works even when keep_prob is other values than 0.5.

# How to do dropout in practice(Inverted Dropout)?

### Forward Propagation 

**Instructions**:
(This is an example from Andrew Ng's class)You would like to shut down some neurons in the first and second layers. To do that, you are going to carry out 4 Steps:
1. In lecture, we dicussed creating a variable $$d^{[1]}$$ with the same shape as 
$$a^{[1]}$$ using `np.random.rand()` to randomly get numbers between 0 and 1. Here, you will use a vectorized implementation, so create a random matrix $$D^{[1]} = [d^{[1](1)} d^{[1](2)} ... d^{[1](m)}] $$ of the same dimension as $$A^{[1]}$$.
2. Set each entry of $$D^{[1]}$$ to be 0 with probability (`1-keep_prob`) or 1 with probability (`keep_prob`), by thresholding values in $$D^{[1]}$$ appropriately. Hint: to set all the entries of a matrix X to 0 (if entry is less than 0.5) or 1 (if entry is more than 0.5) you would do: `X = (X < 0.5)`. Note that 0 and 1 are respectively equivalent to False and True.
3. Set $$A^{[1]}$$ to $$A^{[1]} * D^{[1]}$$. (You are shutting down some neurons). You can think of $$D^{[1]}$$ as a mask, so that when it is multiplied with another matrix, it shuts down some of the values.
4. Divide $$A^{[1]}$$ by `keep_prob`. By doing this you are assuring that the result of the cost will still have the same expected value as without drop-out. (This technique is also called inverted dropout.)

### Backward Propagation 

**Instruction**:
Backpropagation with dropout is actually quite easy. You will have to carry out 2 Steps:
1. You had previously shut down some neurons during forward propagation, by applying a mask $$D^{[1]}$$ to `A1`. In backpropagation, you will have to shut down the same neurons, by reapplying the same mask $$D^{[1]}$$ to `dA1`. 
2. During forward propagation, you had divided `A1` by `keep_prob`. In backpropagation, you'll therefore have to divide `dA1` by `keep_prob` again (the calculus interpretation is that if $$A^{[1]}$$ is scaled by `keep_prob`, then its derivative $$dA^{[1]}$$ is also scaled by the same `keep_prob`).


# What's mini-batch gradient descent? Why use it?

If you take all the training data set to do gradient descent, it will take long time, especially you have a huge datasets. Instead of using all, we can divide the data set evenly into $$K$$, each one with, say, 1000, data points.
Then each time, we use 1000 data points as a mini-batch to move one step of gradient descent.

* if mini-batch size = m, Batch gradient descent, $$(X^{\{1\}},Y^{\{1\}}) = (X, Y)$$
 * Then batch gradient descent might start somewhere and be able to take relatively low noise, relatively large steps. And you could just keep matching to the minimum
* if mini-batch size = 1, Stochastic gradient descent, every example is its own mini-batch.
 * stochastic gradient descent If you start somewhere let's pick a different starting point. Then on every iteration you're taking gradient descent with just a single strain example so most of the time you hit two at the global minimum. But sometimes you hit in the wrong direction if that one example happens to point you in a bad direction. So stochastic gradient descent can be extremely noisy. And on average, it'll take you in a good direction, but sometimes it'll head in the wrong direction as well. As stochastic gradient descent won't ever converge, it'll always just kind of oscillate and wander around the region of the minimum. 
 * But a huge disadvantage to stochastic gradient descent is that you lose almost all your speed up from vectorization. 
* __In practice, batch size is between [1, m]__:
  - get a lot vectorization, faster learning
  - make progress without needing to wait to you process the entire training set.
  
![](/assets/Screen Shot 2017-10-29 at 1.29.40 AM.png)

# How to choose mini-batch size?

* If small data set: just use batch gradient descent
* Typical mini-bath size: 64, 128, 256, 512, 1024
* make sure mini-batch fit in CPU/GPU memory


# What's gradient descent with momentum?

In one sentence, the basic idea is to compute an exponentially weighted average of your gradients, and then use that gradient to update your weights instead.

$$
\begin{aligned} 
   V_{dW} &= \beta_1 * V_{dW} + (1 - \beta_1) * dW \\
   V_{db} &= \beta_1 * V_{db} + {1 - \beta_1} * db \\
   W :&= W - \alpha * V_{dW} \\
   b :&= b - \alpha * V_{db}  
\end{aligned}
$$

Generally, we use $$\beta = 0.9$$, which means we're averaging  over last 10 iteration gradients. 

**What you should remember**:
- Momentum takes past gradients into account to smooth out the steps of gradient descent. It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent.
- You have to tune a momentum hyperparameter $$\beta$$ and a learning rate $$\alpha$$.

# What's RMSprop?
RMSprop, which stands for root mean square prop, that can also speed up gradient descent.

$$
\begin{aligned} 
   S_{dW} &= \beta_2 * S_{dW} + (1 - \beta_2) * (dW)^2  \rightarrow (element\_wise\_squared) \\
   S_{db} &= \beta_2 * S_{db} + {1 - \beta_2} * (db)^2  \rightarrow (element\_wise\_squared)\\
   W :&= W - \alpha * \frac{dW}{\sqrt{S_{dW}}} \\
   b :&= b - \alpha * \frac{db}{\sqrt{S_{db}}}  
\end{aligned}
$$

# What's Adam optimization?

take momentum and RMSprop together.

First, let $$V_{dW} = 0, S_{dW} = 0, V_{db} = 0, S_{db} = 0$$.Then, on iteration t:
Compute $$d_W$$,$$d_b$$ using mini-batch.


$$
\begin{aligned}
&V_{dW} = \beta_1 * V_{dW} + (1 - \beta_1) * dW \\
&V_{db} = \beta_1 * V_{db} + {1 - \beta_1} * db \\
&V_{dW}^{corrected} = \frac{V_{dW}}{1-\beta_1^t}  \rightarrow (bias\_correction)\\
&V_{db}^{corrected} = \frac{V_{db}}{1-\beta_1^t}  \rightarrow (bias\_correction)\\
&S_{dW} = \beta_2 * S_{dW} + (1 - \beta_2) * (dW)^2 \rightarrow (element\_wise\_squared) \\
&S_{db} = \beta_2 * S_{db} + {1 - \beta_2} * (db)^2 \rightarrow (element\_wise\_squared)\\
&S_{dW}^{corrected} = \frac{S_{dW}}{1-\beta_2^t} \rightarrow (bias\_correction) \\
&S_{db}^{corrected} = \frac{S_{db}}{1-\beta_2^t} \rightarrow (bias\_correction) \\
&W:= W - \lambda * \frac{V_{dW}^{corrected}}{\sqrt{S_{dW}^{corrected}}+\epsilon}\\
&b:= b - \lambda * \frac{V_{db}^{corrected}}{\sqrt{S_{db}^{corrected}}+\epsilon}

\end{aligned}
$$

Hyper-parameters:

* $$\alpha$$ nees to be tuned
* $$\beta_1$$: 0.9   $$\rightarrow  (dW)$$ first moment 
* $$\beta_2$$: 0.999  $$\rightarrow (dW^2) $$ second moment

__Adam__: Adaptive moment estimation 


# What are the parameters more important in DLand have to be tuned?

* __learning rate $$\alpha$$__
* __momentum rate $$\beta$$__
* Adam: $$\beta_1= 0.9$$, $$\beta_2=0.999$$ and $$\epsilon$$ 
* number of layers
* number of hidden units
* learning rate decay 
* __mini-batch size $$m$$__ 



# What's batch normalization?

Normalize hidden unit values $$Z^{[l]}$$(before activation function) or $$a^{[l]}$$ (after activation function) so as
to train $$W$$ and $$b$$ faster. Mostly, do $$Z[l]$$. For some specific layer $$l$$, calculate 

$$
\begin{aligned}
 &\mu^{[l]} = \frac{1}{m} \sum_i Z^{[l](i)} \\
 &\sigma^{2[l]} = \frac{1}{m} \sum_i(Z^{[l](i)} - \mu^{[l]})^2 \\
 &Z_{norm}^{[l](i)}  = \frac{Z^{[l](i)}-\mu}{\sqrt{\sigma^{2[l]} + \epsilon}} \\
 &\tilde{Z}^{[l](i)}  = \gamma Z_{norm}^{[l][i]} + \beta
\end{aligned}
$$

Then use $$\tilde{Z}^{[l](i)}$$ instead of $$Z^{[l](i)}$$ for your hidden unit values. BN also has a slight regularization effect. But, if miniBatch Size is large, say, 512, this effect goes away.


Adding  batch norm to a network 
![](/assets/Screen Shot 2017-10-30 at 11.21.20 AM.png)


# Why batch normalization works?

Why batch norm works, is it makes weights, later or deeper than your network, say the weight on layer 10, more robust to changes to weights in earlier layers of the neural network, say, in layer one. 

__Convraince shift__: if you've learned some X to Y mapping, if the distribution of X changes, then you might need to retrain your learning algorithm

# How to do batch norm at test time?

Batch norm processes your data one mini batch at a time, but the test time you might need to process the examples one at a time. Then, how to do that? __Estimate__ $$\mu$$ and $$\sigma^2$$ __using exponentially weighted average across all mini-batchs__.
For example, for the layer $$l$$, you have mini-batch $$X^{\{1\}}$$, $$X^{\{2\}}$$, $$X^{\{3\}}$$, ...., after training each mini-batch, you have the following values $$\mu^{\{1\}[l]}$$, $$\mu^{\{2\}[l]}$$, $$\mu^{\{3\}[l]}$$, .... and $$\sigma^{\{1\}[l]}$$, $$\sigma^{\{2\}[l]}$$, $$\sigma^{\{3\}[l]}$$





























 





