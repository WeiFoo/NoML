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