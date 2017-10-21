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
\frac{\partial}{\partial\theta_j}l\left(\theta\right)=\left(y^{(i)}\frac{1}{h_{\theta}\left(x\right)}-\left(1-y^{(i)}\right)\frac{1}{1-h_{\theta}\left(x\right)}\right)\frac{\partial}{\partial\theta_j}h_{\theta}\left(x\right)
$$
$$
=\left(\frac{y^{(i)}\left(1-h_{\theta}\left(x\right)\right)-\left(1-y^{(i)}\right)h_{\theta}\left(x\right)}{h_{\theta}\left(x\right)\left(1-h_{\theta}\left(x\right)\right)}\right)h_{\theta}\left(x\right)\left(1-h_{\theta}\left(x\right)\right)\frac{\partial}{\partial\theta_j}\theta^Tx
$$

$$
=\left(y^{(i)}-h_{\theta}\left(x\right)\right)x_j
$$

Then, we can update the parameters by SGD:
$$
\theta_j:=\theta_j+a\left(y^{\left(i\right)}-h_{\theta}\left(x^{\left(i\right)}\right)\right)x_{j}^{\left(i\right)}
$$
 
## Solvers:

Given the logistic loss function(-log(x)):
$$
min J(w) = min {-\frac{1}{m}[\sum_{i=1}^{m}y_ilog h_w (x_i) + (1-y_i)log(1-h_w(x_i))]} 
$$
We use $$g$$ and $$H$$ to denote 1st order gradient and Hessian matrix. For a given sample $$y_i$$, we have

$$
g_j = \frac{\partial J(w)} {\partial w_j} = \frac{y^{(i)}}{h_w(x^{(i)})}h_w(x^{(i)})(1-h_w(x^{(i)}))(-x_{j}^{(i)})+(1-y^{(i)})\frac {1}{1-h_w(x^{(i)})}h_w(x^{(i)})(1-h_w(x^{(i)}))x_j^{(i)}=(y^{(i)}-h_w(x^{(i)}))x^{(i)}    
 $$
$$
H_{mn} = \frac {\partial^2 J(w)} {\partial w_m \partial w_n} =h_w(x^{(i)})(1-h_w(x^{(i)}))x^{(i)}_mx^{(i)}_n  
 $$
 
 #### SGD
 
We use the gradient to find the direction to reduce the loss, $$ w_j^{k+1} = w_j^k + \alpha g_j$$. $$k$$ is the iteration number. After each update, we can compare $$J(w^{k+1}) - J(w^k) $$ or $$||w^{k+1}-w^{k}||$$ with some threshold $$\epsilon$$ to determine when to stop.
 
 #### Newtown
 
The basic idea of Newtown method is to do second order Taylor expansion of f(x) around current local optima value, to get the estimates of next optimal value. 
Suppose $$w^{k}$$is the current minimal value,$$
 \varphi (w) = J(w^k) + J'(w^k)(w-w^k)+\frac{1}{2}J''(w^k)(w-w^k)^2 
$$
 
Let $$\varphi'(w) = 0$$, then we get $$w = w^k-\frac{J'(w^k)}{J''(w^k)}$$, then we have the update rule:
$$
   w^{k+1} = w^k - \frac{J'(w^k)}{J''(w^k)} = w^k - H_k^{-1}\cdot g_k
$$
 
In this method, we need a threshold $$\epsilon$$, when $$||g_k|| < \epsilon$$, stop updates. In this method, we also require that the 2nd order derivative of objective function J(w) exists. 


#### BFGS

Issues with Newtown:
* $$H^{-1}_k$$ is expensive, computing intensive. 
* $$H^{-1}_k$$ might be not semi-definite 
Therefore, we need to use other quasi-newton methods, which is to estimate $$H^{-1}_k$$ matrix.

##### Secant condition 

After k+1 iterations, Taylor expansion around $$x_{k+1}$$ is,

$$ 
 f(x) \approx f(x) + f(x_{k+1})'(x-x_{k+1}) + \frac{1}{2}f(x_{k+1})''(x-x_{k+1})^2
$$

Take a derivative, 

$$
 f'(x) = f'(x_{k+1}) +H_{k+1}(x-x_{k+1})
$$

let $$x= x_{k}$$, then we have  

$$
 g_{k+1} - g_k \approx H_{k+1}(x_{k+1}-x_k)
$$

Let $$s_k = x_k - x_{k+1}$$ and $$y_k = g_{k+1} - g_k$$, then we have the secant condition as:

$$
 y_k = H_{k+1}s_k \:\:\:\:\:\:\:\:\:\:(1)
$$

##### Method
To estimate $$H^{-1}_k$$, we use $$B_k \approx H_{k}$$, then we hope that we can use the following rule to update:

$$
B_{k+1} = B_k + \Delta B_k \:\:\:\:\:\:\:\:\:\:(2)

$$
We always set $$B_0 $$ as unit matrix $$I$$, then we suppose the structure of $$\Delta B_k$$ as follows:

$$
\Delta B^k = \alpha uu^T + \beta vv^T
$$

According to the secant condition (1), we have the following function,

$$
y_k = B_ks_k+(\alpha u^Ts_k)u+(\beta v^Ts_k)v
$$
let $$\alpha u^Ts_k = 1$$ and $$\beta v^Ts_k=1$$,  Choosing $$u=y_k$$ and $$v = B_ks_k$$, we can obtain:

$$
\alpha  = \frac{1}{y_k^Ts_k}, \qquad \beta = - \frac{1}{s_k^TB_ks_k}
$$

Finally, we substitute $$\alpha$$ and $$\beta$$ into (2) and get the update equation of $$B_{k+1}$$

$$
B_{k+1} = B_k + \frac{y_ky_k^T}{y_k^Ts_k} - \frac{B_ks_ks_k^TB_k^T}{s_k^TB_ks_k}
$$

##### Algorithm

1. Initialize $$x_0$$ and threshold $$ \epsilon $$, let $$B_0 = I$$, $$k=0$$.
2. Obtain the direction, $$ d_k=-B_k^{-1}g_k$$.
3. Perform a one-dimensional optimizaiton (linear search) to find an acceptable stepsize, $$\lambda_k$$, so $$\lambda_k = \arg\limits_{\lambda} minf(x_k +\lambda*d_k)$$
4. Set $$ s_k = \lambda_k * d_k $$, $$x_{k+1} = x_k + s_k$$.
5. If $$||g_{k+1}||< \epsilon$$, algorithm terminates.
6. Update $$ y_k = g_{k+1} - g_{k}$$
7. Update $$B_{k+1} = B_k +\frac{y_ky_k^T}{y_k^Ts_k} - \frac{B_ks_ks_k^TB_k^T}{s_k^TB_ks_k}$$

Note: in step 2, we have to calculate $$B_k^{-1}$$, which can be done by [Sherman-Morrison formula](https://en.wikipedia.org/wiki/Sherman–Morrison_formula).

 
 
[1][Hessian Matrix of Logistic function](http://personal.psu.edu/jol2/course/stat597e/notes2/logit.pdf) 
[2][BFGS wiki](https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm)
## Can logistic regression work on data that may not be separable by a linear boundary? 

Yes, kernel trick, project data to higher dimension feature space, which might have a hyperplane to separate the data. 


## What's the relationship between logistic regression and gaussian naive bayes?

Under some condition, they learn the same model.

In gaussian naive bayes, Posterior is :

$$
p(y|x)=\frac{P(x|y)P(y)}{\sum P(x|y)P(y)}
$$

Generally, we assume that $$P(x|y)$$ is gaussian distribution, and $$P(y)$$ is polynomial distribution, then the parameters could be estimated by MLE. If we only consider a binary classification problem, then the log of odd's ratio can be written by:

$$
log\frac{P(y=1|x)}{P(y=0|x)}=log\frac{P(x|y=1)}{P(x|y=0)}+log\frac{P(y=1)}{P(y=0)}=-\frac{(x-\mu_1)^2}{2\sigma_1^2}+\frac{(x-\mu_0)^2}{2\sigma_0^2}+\theta_0
$$

if we assume $$\sigma_1=\sigma_0$$, then denominator would be cancelled out, we will get:

$$
log\frac{P(y=1|x)}{P(y=0|x)}=\theta^Tx
$$
Furthermore, we will get:

$$
P(y=1|x)=\frac{e^{\theta^Tx}}{1+e^{e^Tx}}=\frac{1}{1+e^{-\theta^Tx}}
$$

This is the same as logistic regression.

[1] [Chinese reference](https://plushunter.github.io/2017/01/12/机器学习算法系列（3）：逻辑斯谛回归/)

## How to apply logistic regression on multi-classification problem?
* Build binary classifier for each category, one-verus- all
* Softmax


#### A set of independent binary regressions

One fairly simple way to arrive at the multinomial logit model is to imagine, for K possible outcomes, running K-1 independent binary logistic regression models, in which one outcome is chosen as a "pivot" and then the other K-1 outcomes are separately regressed against the pivot outcome.

 $$
 ln\frac{Pr(Y_i = 1)}{P(Y_i = K)} = \beta_1 X_i
 $$ 
 
 $$
ln\frac{Pr(Y_i = 2)}{P(Y_i = K)} = \beta_2 X_i
$$$$
......
$$$$
ln\frac{Pr(Y_i = K-1)}{P(Y_i = K)} = \beta_{K-1} X_i
$$


Note that we have introduced separate sets of regression coefficients, one for each possible outcome. If we exponentiate both sides, and solve for the probabilities, we get:

$$
Pr(Y_i = 1) = Pr(Y_i = K) e^{\beta_1X_i}
$$$$Pr(Y_i = 2) = Pr(Y_i = K) e^{\beta_2X_i}
$$$$
......
$$$$
Pr(Y_i = K-1) = Pr(Y_i = K) e^{\beta_{K-1}X_i}
$$

Using the fact that all K of the probabilities must sum to one, we find:
$$
Pr(Y_i = K) = 1 - \sum_{k=1}^{K-1}Pr(Y_i = K)e^{\beta_kX_i} \Rightarrow Pr(Y_i = K) = \frac{1}{1+\sum_{k=1}^{K-1}e^{\beta_kX_i}}
$$

We can use this to find the other probabilities:
$$
 Pr(Y_i = 1) = \frac{e^{\beta_1Xi}}{1+ \sum_{k=1}^{K-1}e^{\beta_kX_i}}
$$$$
Pr(Y_i = 2) = \frac{e^{\beta_2Xi}}{1+ \sum_{k=1}^{K-1}e^{\beta_kX_i}}
$$$$
......$$$$
Pr(Y_i = K-1) = \frac{e^{\beta_{K-1}Xi}}{1+ \sum_{k=1}^{K-1}e^{\beta_kX_i}}
$$

The fact that we run multiple regressions reveals why the model relies on the assumption of independence of irrelevant alternatives described above.


[1] [https://en.wikipedia.org/wiki/Multinomial_logistic_regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)
#### Sigmox  

More:

Properties of Softmax:

* The calculated probabilities will be in the range of 0 to 1.
* The sum of all the probabilities is equals to 1.

When using Softmax, we have the following probability model
$$
P(y=i|x,\theta)=\frac{^{e^{\theta^T_ix}}}{\sum_j^K e^{\theta_j^Tx}}
$$

We use the following rule to classify our prediction:

$$
y^*=argmax_iP(y=i|x,\theta )
$$

The loss function is

$$
J(\theta)=-\frac{1}{N}\sum_i^N\sum_j^KP(y_i=j)log\frac{e^{\theta_i^Tx}}{\sum e^{\theta_k^Tx}}
$$

## Compare logistic regression with SVM

Similarity:

* classification algo
* supervised learning
* discriminative model
* both can be used for non-leaner classification by kernel trick
* both objective is to find a hyperplane
* 都能减少离群点的影响?? I don't understand

Difference:
* loss functions: SVM: hinge loss, logistic regression: cross-entropy loss
* when estimating the parameters, all samples get involved in LR while SVM only use the support vector samples. Since all samples are used for estimating parameters, we need to consider imbalance issue for each category.
* LR model probability, SVM model hyperplan
* LR is statistical method, SVM is geometrical method
* LR avoid the effects of samples which are far away from the separating plane, while SVM only use support vector to mitigate the effects of samples which are far from hyperplane 
* SVM is more sensitive to outliers since it only require support vectors for training while LR uses all training samples.


[1] [Chinese reference1](https://plushunter.github.io/2017/01/12/机器学习算法系列（3）：逻辑斯谛回归/)

[2] [Chinese reference2 浅析Logistics Regression](https://chenrudan.github.io/blog/2016/01/09/logisticregression.html)



## Compare logistic regression with Naive bayes

Similarity:

* classification algo
* supervised learning
* when the conditional probability$$P(X|Y=c_k)$$ in naive Bayes is assumed to be Gaussian IID, then the expression of $$P(Y=1|X)$$ is the same as logistic regression.

Difference:
* Logistctic regression is discriminative model, it learns $$P(y|x)$$ while Naive Bayes is Discriminative, it learns $$P(x,y)$$.
* The former need iterative calculation, the latter one doesn't 
* When data is limited, Naive Bayes is better then Logistic regression; Otherwise, Logistic regression is better than Naive Bayes when large amount of data is available.
* Since Naive Bayes assumes Gaussian IID $$P(X|y)$$, which means features are independent; if this does not hold, then Naive Bayes is not better than LR.

[1][Chinese reference.  浅析Logistics Regression](https://chenrudan.github.io/blog/2016/01/09/logisticregression.html)
 

















