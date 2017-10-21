# Decision Tree

### Bacics

* Non-parametric, supervised learning algorithms
* Given the training data, a decision tree algorithm divides the feature space into regions. For inference, we first see which region does the test data point fall in, and take the mean label values (regression) or the majority label value (classification).
* Construction: top-down, chooses a variable to split the data such that the target variables within each region are as homogeneous as possible. Two common metrics: __gini impurity__ or __information gain__, won't matter much in practice.
* Advantage: simply to understand & interpret, mirrors human decision making
* Disadvantage:
  * can __overfit easily__ (and generalize poorly)if we don't limit the depth of the tree
  * can be __non-robust__: A small change in the training data can lead to a totally different tree
  * __instability__: sensitive to training set rotation due to its orthogonal decision boundaries
  
  [https://github.com/ShuaiW/ml-interview#Decision tree](https://github.com/ShuaiW/ml-interview#SVM)
  

### Least Square Regression Tree

__Input__: Training data D
__Output__: regression tree f(x)
__Splitting Criteria__: MSE

1. Obtain the best optimal variable(feature) $$j$$ and splitting pint $$s$$, according to:
$$
\min_{j,s}\left[\min_{c_1}\sum_{x_i\in R_1(j,s)}(y_i-c_1)^2+\min_{c_2}\sum_{x_i\in R_2(j,s)}(y_i-c_2)^2\right] \:\:\:\:\: (1)
$$

Iterate all variable $$j$$ and for a given splitting point $$s$$, we need to get a pair of $$(j,s)$$ to minimize formula (1)

2. Divide the training data D with $$(j,s)$$ and output the values
  $$
  R_1(j,s)=\{x|x^{(j)}\le s\} \quad ,\quad R_2(j,s)=\{x|x^{(j)}\gt s\}
  $$
  
  $$
  \hat{c}_m=\frac{1}{N_m}\sum_{x_i\in R_m(j,s)}y_i, x\in R_m, m=1,2
  $$
  
3. Repeat step 1 and 2  to divide regions $$R_1$$ and $$R_2$$ until stopping criteria is satisfied.

4. Divide the input space into $$M$$ regions$$R_1, R_2,...R_M$$, output the regression tree:
$$
f(x)=\sum_{m=1}^Mc_mI(x\in R_m)
$$
  


