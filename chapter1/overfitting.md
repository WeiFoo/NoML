# Overfitting/Underfitting


# What's Bias-variance trade-off?

__Bias__: Difference between the expected(or average) prediction of our model and the correct value which we are trying  to predict. larger bias means underfitting and the model we got is simple.
__Variance__:

[1] [Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html)

[2][Andrew Ng's CS229 notes on learning theory](http://cs229.stanford.edu/notes/cs229-notes4.pdf)



## **What's overfitting?**

A: Overfitting refers to a model that models the training data too well. Overfitting happens when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data.



E: Overfitting is more likely with nonparametric and nonlinear models that have more flexibility when learning a target function. As such, many nonparametric machine learning algorithms also include parameters or techniques to limit and constrain how much detail the model learns. For example, decision trees are a nonparametric machine learning algorithm that is very flexible and is subject to overfitting training data. This problem can be addressed by pruning a tree after it has learned in order to remove some of the detail it has picked up.

R:[ https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms](https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/)

## Why overfitting happens?

Overfitting occurs when a model is excessively complex, such as having too many [parameters](https://en.wikipedia.org/wiki/Parameter) relative to the number of observations. Or the number of training data is not large enough!

R: [https://en.wikipedia.org/wiki/Overfitting\](https://en.wikipedia.org/wiki/Overfitting)

## How to avoid overfitting?

Early Stopping, Data argumentation\(numbers of parameter is much smaller than the data \), cross-validation,  Regularization (L1, L2\), Dropout... 

R:  [https://www.quora.com/What-are-ways-to-prevent-over-fitting-your-training-set-data-Is-it-possible-to-test-if-you-have-done-so](https://www.quora.com/What-are-ways-to-prevent-over-fitting-your-training-set-data-Is-it-possible-to-test-if-you-have-done-so)


## What's early stopping?


Early stopping is a form of regularization used to avoid overfitting when training a learner with an iterative method, such as gradient descent.Divide the training data into new_train and validation data.

After each epoch, evaluate the accuracy(error) on the validation data, if the accuracy doesn't improve over N epochs, then we that the accuracy won't increase accuracy. N can be 10, 20,.....

Early stopping rules provid guidance as to how many iterations can be run before the learner begins to over-fit. 

R: [https://en.wikipedia.org/wiki/Early_stopping](https://en.wikipedia.org/wiki/Early_stopping)

## What's dropout?

At each training stage, individual nodes are either "dropped out" of the net with probability 1−p or kept with probability p, so that a reduced network is left; incoming and outgoing edges to a dropped-out node are also removed. Only the reduced network is trained on the data in that stage. The removed nodes are then reinserted into the network with their original weights.

The technique seems to reduce node interactions, leading them to learn more robust features that better generalize to new data.

R:[1] [https://en.wikipedia.org/wiki/Convolutional_neural_network#Dropout](https://en.wikipedia.org/wiki/Convolutional_neural_network#Dropout)
  [2] [http://cs231n.github.io/neural-networks-2/](http://cs231n.github.io/neural-networks-2/)














