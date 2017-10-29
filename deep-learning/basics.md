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

