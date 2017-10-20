# Logistic Regreesion

**Logistic function or Sigmoid function**:  
$$g(x) = \frac{1}{1+e^{-x}}$$,  which will take any $$x \in R$$, and output a value between 0 and 1.

**Derivative of sigmoid function:**

$$g^{\prime} = g(x)*(1-g(x))$$

If we have $$ t = \beta_0 + \beta_1 x$$,  then the logistic function can be written as: $$F(x) = \frac{1}{1+ e^{-(\beta_ 0 + \beta_1 x)}}$$. Note that the $$F(x)$$ is interpreted as the probability of the dependent variable equaling a "success" or "case" rather than a failure or non-case.

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

## Why logistic regression use sigmoid function?



