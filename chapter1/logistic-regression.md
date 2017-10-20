  # Logistic Regreesion

**Logistic function or Sigmoid function**:  
$$g(x) = \frac{1}{1+e^{-x}}$$,  which will take any $$x \in R$$, and output a value between 0 and 1.

**Derivative of sigmoid function:**

$$g^{\prime} = g(x)*(1-g(x))$$

If we have $$ t = \beta_0 + \beta_1 x$$,  then the logistic function can be written as: $$F(x) = \frac{1}{1+ e^{-(\beta_ 0 + \beta_1 x)}}$$. Note that the $$F(x)$$ is interpreted as the probability of the dependent variable equaling a "success" or "case" rather than a failure or non-case.

[1] [Reference In Chinese](https://plushunter.github.io/2017/01/12/机器学习算法系列（3）：逻辑斯谛回归/)

## Why logistic regression is a linear model?

The logit of a number p between 0 and 1 is given by $$logit(p)= ln(\frac{p}{1-p})$$. Therefore the logit of F\(x\) can be written as  
$$logit(F(x)) = ln(\frac{F(x)}{(1-F(x))}) = \beta_0 + \beta_1 x....$$

It's called a generalized linear model **not because** the estimated probability of the response event is linear, **but because the logit of the estimated probability** response is a linear function of parameters.

More generally, the Generalized Linear Model is of the form   
$$g(\mu_i) = \beta_0 + \beta_1x+\beta_2x^2.....$$. where $$\mu$$ is the expected value of the response given the covariates.

\[1\][https://en.wikipedia.org/wiki/Logit](https://en.wikipedia.org/wiki/Logit)  
\[2\][https://en.wikipedia.org/wiki/Logistic\_regression](https://en.wikipedia.org/wiki/Logistic_regression)  
\[3\][https://stats.stackexchange.com/questions/88603/why-is-logistic-regression-a-linear-model](https://stats.stackexchange.com/questions/88603/why-is-logistic-regression-a-linear-model)

## Why isn't logistic regression called logistic classification?

Logistic regression was called regression long before terms like supervised learning come along and regreresion does not, in general, imply continuous outcomes. 

Logistic regression is not a classification algorithm on its own. It's only a classification algorithm in combination with a decision rule (i.e., if g(x)>0.5 then y=1 else y=0) that makes LR as a classifier.

In addition, logistic regreesion technically is on continuous variable: It is regression on the logit of the event, that is $$log(\frac{p}{1-p})$$, where p is the probability of the event. Logistic regression is a regression model because it estimates the probability of class membership as a (transformation of a) multilinear function of the features.


[1] [https://www.quora.com/Why-is-logistic-regression-called-regression-if-it-doesnt-model-continuous-outcomes](https://www.quora.com/Why-is-logistic-regression-called-regression-if-it-doesnt-model-continuous-outcomes)

[2] https://stats.stackexchange.com/questions/127042/why-isnt-logistic-regression-called-logistic-classification

 

## Why logistic regression use sigmoid function? Can we use others?
Yes, we can use others. any non-linear function will do. But sigmoid function has some beautiful properties.

* Sigmoid function is founded between 0 and 1, and its derivative is easy to calculate.
* It's a simple way to introduce  non-linearity to the model 

Other motivations:
Sigmoid outputs the conditional prob of the prediction. The "odds ratio" is defined as $$\frac{p}{1-p}$$, the ratio between the prob an event occurs and the prob the event doesn't occur. If you take the natural log of this odds' ratio, we get **logit function** 

$$
logit(p(y=1|X))=log(\frac{p}{1-p})
$$

Let's use this log-transformation to model the relationship between our variable and the target variable,
$$
 logit(p(y=1|X))=log(\frac{p}{1-p}) = \beta_0 + \beta_1x_1+....
$$

We're interested in $$p(y=1|X)$$, then take the inverse of this logit function , we get the logistic sigmoid 
$$
logit^{-1}(p(y=1|x))=\frac{1}{1+e^{-{\beta_0 + \beta_1x_1+....}}}
$$

[1] https://www.quora.com/Logistic-Regression-Why-sigmoid-function


## How to update(estimate) logistic regression parameters?[THIS IS LONG]
**Sort answer: ** MLE, SDG, L-BFGS, ADMM....???

**Long answer:** 

Suppose the samples are generated from bernoulli process and the results(0/1) will follow Bernoulli distribution. Suppose the prob of 1 is $$h_\theta(x)$$, then prob of 0 is 1-$$h_\theta(x)$$.

For the i-th sample, the probability can be written as:

$$
P(y^{(i)}=1|x^{(i)};\theta )=h_\theta{(x^{(i)})} 
$$ $$
P(y^{(i)}=0  |x^{(i)};\theta )=1- h_\theta{(x^{(i)})}
$$

Then combine them together, the prob of correct prediction on i-th sample is:
$$
P(y^{(i)}|x^{(i)};\theta)=(h_\theta(x^{(i)})^{y(i)})(1-h_\theta(x^{(i)}))^{1-y(i)}
$$

Since we assume that all the samples are generated IID, then for all N samples, the probability distribution can be expressed as:

 $$
 P\left(Y|X;\theta\right)=\prod_{i=1}^N{\left(h_{\theta}\left(x^{\left(i\right)}\right)^{y^{\left(i\right)}}\left(1-h_{\theta}\left(x^{\left(i\right)}\right)^{1-y^{\left(i\right)}}\right)\right)}
 $$ 
 
 $$
 L\left(\theta\right)=P\left({Y}|X;\theta\right) 
 $$
To estimate the parameters $$\theta$$, we take the log of the above expression, then we get log-liklihood function:
$$
l\left(\theta\right)=\sum_{i=1}^N{\log l\left(\theta\right)}

=\sum_{i=1}^N{y^{\left(i\right)}\log\left(h_{\theta}\left(x^{\left(i\right)}\right)\right)+\left(1-y^{\left(i\right)}\right)\log\left(1-h_{\theta}\left(x^{\left(i\right)}\right)\right)}
$$

You might feel familiar with this. Right! Maximizing the log likelihood is actually minimizing the cross entropy error.
For now, we just ignore the summation term and consider very single parameter $$\theta_j$$, we take the derivative with respective to $$\theta_j$$

$$
\frac{\partial}{\partial\theta_j}l\left(\theta\right)=\left(y\frac{1}{h_{\theta}\left(x\right)}-\left(1-y\right)\frac{1}{1-h_{\theta}\left(x\right)}\right)\frac{\partial}{\partial\theta_j}h_{\theta}\left(x\right)
$$
$$
=\left(\frac{y\left(1-h_{\theta}\left(x\right)\right)-\left(1-y\right)h_{\theta}\left(x\right)}{h_{\theta}\left(x\right)\left(1-h_{\theta}\left(x\right)\right)}\right)h_{\theta}\left(x\right)\left(1-h_{\theta}\left(x\right)\right)\frac{\partial}{\partial\theta_j}\theta^Tx
$$

$$
=\left(y-h_{\theta}\left(x\right)\right)x_j
$$

Then, we can update the parameters by:
$$
\theta_j:=\theta_j+a\left(y^{\left(i\right)}-h_{\theta}\left(x^{\left(i\right)}\right)\right)x_{j}^{\left(i\right)}
$$

## Can logistic regression work on data that may not be separable by a linear boundary? 

Yes, kernel trick, project data to higher dimension feature space, which might have a hyperplane to separate the data. 



















