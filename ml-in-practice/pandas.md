# Pandas

Before this, strongly suggest practice [pandas 10 minutes test drive](https://pandas.pydata.org/pandas-docs/stable/10min.html)

### Create train/test data in pandas

#### Method 1
```python
In [11]: df = pd.DataFrame(np.random.randn(100, 2))
In [12]: msk = np.random.rand(len(df)) < 0.8
In [13]: train = df[msk]
In [14]: test = df[~msk]
```

#### Method 2
```python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)
## this will return pd.dataframe
```
### Convert Pandas dataframe to numpy array.

```python

(Pdb) x = pd.DataFrame(np.random.rand(4,2))
(Pdb) x
          0         1
0  0.437858  0.193156
1  0.672929  0.634040
2  0.616899  0.449750
3  0.691262  0.083987
(Pdb) x.values
array([[ 0.43785847,  0.19315602],
       [ 0.67292882,  0.63403955],
       [ 0.61689947,  0.4497496 ],
       [ 0.69126155,  0.08398693]])
(Pdb) type(x.values)
<type 'numpy.ndarray'>

```
or you can do ```np.array(x.values.tolist())```. Note that in pandas, string is not the same as numpy, it's stored as object.As you noticed, attempting to coerce a python string into a fixed-with numpy string won't work in pandas. Instead, it always uses native python strings, which behave in a more intuitive way for most users. more at [https://stackoverflow.com/questions/34881079/pandas-distinction-between-str-and-object-types](https://stackoverflow.com/questions/34881079/pandas-distinction-between-str-and-object-types)




#  



