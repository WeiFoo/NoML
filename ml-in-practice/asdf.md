# Small tricks

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
# 

#  



