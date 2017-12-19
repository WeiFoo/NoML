## 2. Categorical and Ordinal Features

### 2.1 Label Encoding, maps categories into numbers
- Alphabetical (sorted)
  - [S,C,Q] -> [2, 1, 3]
  - sklearn.preprocessing.[LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
- Order of appearance
  - [S,C,Q] -> [1, 2, 3]
  - Pandas.[factorize](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.factorize.html)

### 2.2 Frequencey encoding, maps categories to their frequencies.
[S,C,Q] -> [0.5, 0.3, 0.2]
- No duplicated frequencies. 
```python
encoding = titanic.groupby(‘Embarked’).size() 
encoding = encoding/len(titanic) 
titanic[‘enc’] = titanic.Embarked.map(encoding)
```

- Has duplicated frequencies.
Categorization after common frequency encodings.
```python
from scipy.stats import rankdata
```

### 2.3 One-hot Encoding
- pandas.[get_dummies](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)

```python
>>> import pandas as pd
>>> s = pd.Series(list('abca'))
>>> pd.get_dummies(s)
   a  b  c
0  1  0  0
1  0  1  0
2  0  0  1
3  1  0  0
```

- sklearn.preprocessing.[OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
```python
>>> from sklearn.preprocessing import OneHotEncoder
>>> enc = OneHotEncoder()
>>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  
OneHotEncoder(categorical_features='all', dtype=<... 'numpy.float64'>,
       handle_unknown='error', n_values='auto', sparse=True)
>>> enc.n_values_
array([2, 3, 4])
>>> enc.feature_indices_
array([0, 2, 5, 9])
>>> enc.transform([[0, 1, 1]]).toarray()
array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.]])
```

### 2.4 Sumarize
- Label and Frequency encodings are often used for *tree-based models*.
- *One-hot encoding* is often used for *non-tree-based models*.
- Interactions of categorical features can help linear models
and KNN
