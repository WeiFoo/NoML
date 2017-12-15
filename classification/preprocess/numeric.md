## 1. Numeric Features

### 1.1 Scaling
First of all,
- Tree-based models doesn't depend on scaling
- Non-tree-based models hugely depend on scaling.

- Normalization, to [0,1]
  - sklearn.preprocessing.[MinMaxScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler)
- Standardization, to mean=0, std=1
  - sklearn.preprocessing.[StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)

### 1.2 Remove Outliers 
- np.[percentile](https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.percentile.html)
  - Compute the qth percentile of the data along the specified axis.
- np.[clip](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.clip.html)
  - Given an interval, values outside the interval are clipped to the interval edges. 
  For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
```python
UPPERBOUND, LOWERBOUND = np.percentile(x, [1,99])
y = np.clip(x, UPPERBOUND, LOWERBOUND)
```

### 1.3 Rank - sets spaces between sorted values to be equal.
scipy.stats.[rankdata](https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.stats.rankdata.html)
```python
>>> from scipy.stats import rankdata
>>> rankdata([0, 2, 3, 2])
array([ 1. ,  2.5,  4. ,  2.5])
>>> rankdata([0, 2, 3, 2], method='min')
array([ 1.,  2.,  4.,  2.])
>>> rankdata([0, 2, 3, 2], method='max')
array([ 1.,  3.,  4.,  3.])
>>> rankdata([0, 2, 3, 2], method='dense')
array([ 1.,  2.,  3.,  2.])
>>> rankdata([0, 2, 3, 2], method='ordinal')
array([ 1.,  2.,  4.,  3.])
```

### 1.4 Other useful transformations.
Often helps non-tree-based methods, especially neural networks.
- $$np.log(1+1)$$
- $$np.sqrt(x + 2/3)$$

### 1.5 Feature generation is powered by 1) Prior knowledge, Exploratory data analysis.
