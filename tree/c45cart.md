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

__Input__: Training data $$D$$
__Output__: regression tree $$f(x)$$
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


### Classification Tree


#### Gini index

Suppose we have $$K$$ classes in a classification problem, $$p_k$$ denotes the probably that the samples with label $$k$$, then the Gini index is defined as:
$$
\text{Gini}(p)=\sum_{k=1}^Kp_k(1-p_k)=1-\sum_{k=1}^Kp_k^2
$$

For a given sample set $$D$$, the Gini index is:

$$
 Gini(D) = 1- \sum_{k=1}^{K}(\frac{|C_k|}{|D|})^2
 $$
where $$K$$ is the total number of classes and $$C_k$$ is the number of samples belong to class $$K$$.

If we divide $$D$$ into $$D_1$$ and $$D_2$$ based on some value $$a$$ of feature $$A$$, then the average Gini index is:

$$
\text{Gini}(D,A)=\frac{|D_1|}{|D|}\text{Gini}(D_1)+\frac{|D_2|}{|D|}\text{Gini}(D_2)\tag{5.25}
$$

Similar to entropy, the higher value of Gini index, the more uncertain of the set it is. 
[1][http://www.ke.tu-darmstadt.de/lehre/archiv/ws0809/mldm/dt.pdf](http://www.ke.tu-darmstadt.de/lehre/archiv/ws0809/mldm/dt.pdf)

[2] [《统计学习方法》笔记 (七) - 决策树(下)](http://daniellaah.github.io/2017/Statistical-Learning-Notes-Chapter5-DecisionTree-2.html)

#### Entropy/Information Gain

  


