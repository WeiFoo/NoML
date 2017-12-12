# Decision Tree

### Bacics

* Non-parametric, supervised learning algorithms
* Given the training data, a decision tree algorithm divides the feature space into regions. For inference, we first see which region does the test data point fall in, and take the mean label values (regression) or the majority label value (classification).
* Construction: top-down, chooses a variable to split the data such that the target variables within each region are as homogeneous as possible. Two common metrics: __gini impurity__ or __information gain__, won't matter much in practice.
* Advantage: simply to understand & interpret, mirrors human decision making
* Disadvantage:
  * can __overfit easily__ (and generalize poorly)if we don't limit the depth of the tree
  * can be __non-robust__: a small change in the training data can lead to a totally different tree
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


#### Gini index(Only used by CART)

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
\text{Gini}(D,A)=\frac{|D_1|}{|D|}\text{Gini}(D_1)+\frac{|D_2|}{|D|}\text{Gini}(D_2)
$$

Similar to entropy, the higher value of Gini index, the more uncertain of the set it is. Therefore, we want to minimize Gini index.
[1][http://www.ke.tu-darmstadt.de/lehre/archiv/ws0809/mldm/dt.pdf](http://www.ke.tu-darmstadt.de/lehre/archiv/ws0809/mldm/dt.pdf)

[2] [《统计学习方法》笔记 (七) - 决策树(下)](http://daniellaah.github.io/2017/Statistical-Learning-Notes-Chapter5-DecisionTree-2.html)

[3][https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)

#### Entropy/Information Gain(Used by C4.5)

Given a set of samples $$S$$, Entropy is defined as below:
$$
E(S)= -\sum_{i=1}^{K} p_i log(p_i) 
$$
where $$p_i$$ represents the percentage of each class present in the child node that results form a split in the tree. It generally shows the amount of unorderedness in the class distribution of $$S$$.
When we split the whole set $$S$$ into $$S_i$$ by feature $$A$$,  the  weighted average over all sets reultsing form the silage is:
$$
I(S,A)  = \sum_{i}\frac{|S_i|}{|S|}E(S_i)
$$

Information gain is defined as information gain = Entropy(parent) - Weighted Sum of Entropy(Children),

$$
IG(S, A) = E(S) - I(S,A) = E(S) - \sum\frac{|S_i|}{|S|}E(S_i)
$$

Maximize the information gain is equivalent to minimize the average entropy, because $$E(S)$$ is constant for all attributes.


#### Information Gain Ratio
The main issue with information gain is that it biases the decision tree against the attributes with a large number of distinct values. For example, the creditcard number attributes of a customer. Here we introduce intrinsic information of a split(How much information do we need to tell which branch an instance belongs to). It's defined as follows:

$$
IntI(S,A) = -\sum_{i}\frac{|S_i|}{|S|} log\frac{|S_i|}{|S|}
$$
Then we can see that attributes with higher intrinsic information are less useful. To reduce the information gain's bias towards multi-valued attributes, we take the number and size of branches into account when choosing an attribute, i.e, corrects the information gain by taking the intrisic information of split into account.
The information gain ratio can be defined as:

$$
GR(S,A) = \frac{IG(S,A)}{IntI(S,A)}
$$

The higher, the better.
[1] [http://www.ke.tu-darmstadt.de/lehre/archiv/ws0809/mldm/dt.pdf](http://www.ke.tu-darmstadt.de/lehre/archiv/ws0809/mldm/dt.pdf)





#### Compare Gini index, information gain  and gain ratio

* Information gain: based towards multivalued attributes.
* Gain ratio: prefer unbalanced splits in which on partition is much smaller than the other.
* Gini Index: 
  * based towards multivalued attributes.
  * favor equal-sized partitions  
  
[1][http://www.inf.unibz.it/dis/teaching/DWDM/slides2011/lesson5-Classification-2.pdf](http://www.inf.unibz.it/dis/teaching/DWDM/slides2011/lesson5-Classification-2.pdf)


#### CART

1. Given a set of training data $$D$$,  calculate Gini index (or IG) on all features. 
  * For each feature $$A$$, we calculate Gini index if we divide $$D$$ into $$D_1$$ and $$D_2$$ when $$A=a$$  
2. Select a pair of feature and split point  $$(A,a)$$ which has minimum Gini diix as the best feature and split.

3. Divide $$D$$ into $$D_1$$ and $$D_2$$

4. repeat step 1 and step 2 on $$D_1$$ and $$D_2$$ until stopping criteria is satisfied.

5. return CART tree.

### CART pruning 

#### Reduced error pruning

One of the simplest forms of pruning is reduced error pruning. Starting at the leaves, each node is replaced with its most popular class. If the prediction accuracy is not affected then the change is kept. While somewhat naive, reduced error pruning has the advantage of simplicity and speed.

[1][https://en.wikipedia.org/wiki/Pruning_%28decision_trees%29](https://en.wikipedia.org/wiki/Pruning_%28decision_trees%29)

#### Cost complexity pruning
1. Generate a series of trees $$T_0, T_1, T_2,....T_n$$ where $$T_0$$is the initial tree and $$T_n$$ is the root alone.
2. At step $$i$$, the tree is created by revmoing a subtree from  tree $$i-1$$ and replacing it with a leaf node with value chosen(majority vote) as in the tree building algorithm.
  * Define the error rate of the tree $$T$$ over data sets $$S$$ as $$err(T, S)$$. The subtree that minimmizes
  $$
    \frac{err(prune(T,t),S)- err(T,S)}{|leaves(T)|-|leaves(prune(T,t))|}
  $$
is chosen for removal. The function $$prune(T,t)$$ defines the tree gotten by pruning the subtree $$t$$ from the tree $$T$$. Once the series of trees has been created, the best tree is chosen by cross-valididation on training data.

Notes: 
* Loss cost of tree T, is defined as:
$$
 err_a(T) = err(T) + a|T|   \:\:\: (2)
$$
* Loss cost of tree T-t, is defined as:

$$
err_a(T,t) = err(T,t)+ a|T-t|  \:\:\: (3)
$$
where $$t$$ is subtree, $$|T-t|$$ is the number of leaves after pruning $$t$$ subtree. IF let (2)=(3), we get:
$$
a = \frac{err(T,t)- err(T)}{|T|-|T-t|}
$$


[1][http://daniellaah.github.io/2017/Statistical-Learning-Notes-Chapter5-DecisionTree-2.html](http://daniellaah.github.io/2017/Statistical-Learning-Notes-Chapter5-DecisionTree-2.html)
[2][https://en.wikipedia.org/wiki/Pruning_%28decision_trees%29](https://en.wikipedia.org/wiki/Pruning_%28decision_trees%29)


  


