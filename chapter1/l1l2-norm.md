# L1/ L2 Regularization

## What's L1 and L2 regularization

L2 is the sum of the sure of the weights
L1 is the sum of the absolute values.

## What's the difference between L1 and L2?
![](http://www.chioka.in/wp-content/uploads/2013/12/L1-vs-L2-properties-regularization.png)

L2 has unique solutions and L1 have multiple solutions. see the following picture.

![](http://www.chioka.in/wp-content/uploads/2013/12/L1-norm-and-L2-norm-distance.png)

R: [http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/](http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/)
 
 
## Why L1 regularization can generate sparsity?
We can use this graph is explain. We have two parameters,theta1 and theta2.
The error value is the same for each blue circle. Regularization terms show in yellow. The first point the blue lines meet yellow line is the optimal solution.
You can see that, for L1, the meeting point mostly happens at the corner while L2 doesn't have corner. ![](/assets/Screen Shot 2017-10-19 at 5.57.54 PM.png)


![](https://pic4.zhimg.com/v2-648584bcfaa1020d62861208775462df_b.png)

## Why L1 is not stable?

![](https://pic1.zhimg.com/v2-e8734136ff4da41b748f16e514971aa0_b.png)


## What's the L1 and L2 loss?

L1 loss is least sum of absolute deviation
L2 loss is least sum of square of the difference tween the target value and the estimated values.


## What's the difference between L1 and l2 as a loss function?

![](http://www.chioka.in/wp-content/uploads/2013/12/L1-vs-L2-properties-loss-function.png)

R: [http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/](http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/)

## Why L1 is more robust?

Least absolute deviations is robust in that **it is resistant to outliers in the data**. LAD gives **equal emphasis to all observations**, in contrast to OLS which, by squaring the residuals, **gives more weight to large residuals**, that is, outliers in which predicted values are far from actual observations. This may be helpful in studies where outliers do not need to be given greater weight than other observations. If it is important to give greater weight to outliers, the method of least squares is a better choice.

R: [https://en.wikipedia.org/wiki/Least_absolute_deviations](https://en.wikipedia.org/wiki/Least_absolute_deviations)







