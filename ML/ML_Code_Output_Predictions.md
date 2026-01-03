# ML Code Output Prediction Problems (30 Questions)

## 📘 Instructions
Predict the output or result of each code snippet **without running it**. These are common exam patterns!

**Scoring**: 1 mark per correct answer (Total: 30 marks)

---

# Section 1: NumPy & Array Operations (10 Questions)

## Q1. What is the output?
```python
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)
```

**A)** `[5 7 9]`  
**B)** `[[1,2,3], [4,5,6]]`  
**C)** Error  
**D)** `[1 2 3 4 5 6]`

**Answer**: A  
**Explanation**: Element-wise addition: [1+4, 2+5, 3+6] = [5, 7, 9]

---

## Q2. What is the shape?
```python
import numpy as np
x = np.zeros((3, 4, 5))
print(x.shape)
```

**A)** `(3, 4, 5)`  
**B)** `(3, 4)`  
**C)** `60`  
**D)** `(5, 4, 3)`

**Answer**: A  
**Explanation**: `zeros((3,4,5))` creates 3D array with shape (3, 4, 5)

---

## Q3. What prints?
```python
import numpy as np
arr = np.array([[1, 2], [3, 4]])
print(arr[1, 0])
```

**A)** `1`  
**B)** `2`  
**C)** `3`  
**D)** `4`

**Answer**: C  
**Explanation**: `arr[1, 0]` = row 1, column 0 = 3

---

## Q4. Output?
```python
import numpy as np
a = np.array([1, 2, 3, 4])
print(a[a > 2])
```

**A)** `[3 4]`  
**B)** `[1 2]`  
**C)** `[True True]`  
**D)** Error

**Answer**: A  
**Explanation**: Boolean indexing returns elements where condition is True: [3, 4]

---

## Q5. Result?
```python
import numpy as np
x = np.arange(6).reshape(2, 3)
print(x.T.shape)
```

**A)** `(2, 3)`  
**B)** `(3, 2)`  
**C)** `(6,)`  
**D)** `(1, 6)`

**Answer**: B  
**Explanation**: Transpose swaps dimensions: (2,3) → (3,2)

---

## Q6. What is the output?
```python
import numpy as np
a = np.array([1, 2, 3])
b = a
b[0] = 99
print(a[0])
```

**A)** `1`  
**B)** `99`  
**C)** `[99 2 3]`  
**D)** Error

**Answer**: B  
**Explanation**: `b = a` creates reference (not copy!). Changing b changes a. Use `a.copy()` for independent copy.

---

## Q7. Output?
```python
import numpy as np
print(np.sum([1, 2, 3, 4]))
```

**A)** `[1 2 3 4]`  
**B)** `10`  
**C)** `2.5`  
**D)** `4`

**Answer**: B  
**Explanation**: `sum([1,2,3,4])` = 1+2+3+4 = 10

---

## Q8. Result?
```python
import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(np.mean(arr, axis=0))
```

**A)** `[2.5 3.5 4.5]`  
**B)** `[2. 5.]`  
**C)** `3.5`  
**D)** `[1.5 2.5 3.5]`

**Answer**: A  
**Explanation**: `axis=0` means along rows. Mean of columns: [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]

---

## Q9. What prints?
```python
import numpy as np
x = np.array([1, 2, 3, 4, 5])
print(x[-2])
```

**A)** `2`  
**B)** `4`  
**C)** `5`  
**D)** Error

**Answer**: B  
**Explanation**: Negative indexing: -1=last, -2=second to last = 4

---

## Q10. Output?
```python
import numpy as np
a = np.array([1, 2])
b = np.array([[3], [4]])
print((a * b).shape)
```

**A)** `(2,)`  
**B)** `(2, 1)`  
**C)** `(2, 2)`  
**D)** Error

**Answer**: C  
**Explanation**: Broadcasting: (2,) and (2,1) → broadcast to (2,2). Result shape: (2,2)

---

# Section 2: Pandas Operations (10 Questions)

## Q11. What is the output?
```python
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.shape)
```

**A)** `(3, 2)`  
**B)** `(2, 3)`  
**C)** `6`  
**D)** `(3,)`

**Answer**: A  
**Explanation**: DataFrame has 3 rows, 2 columns → (3, 2)

---

## Q12. Result?
```python
import pandas as pd
s = pd.Series([1, 2, 3, 4, 5])
print(s[s > 3].sum())
```

**A)** `9`  
**B)** `15`  
**C)** `4`  
**D)** `5`

**Answer**: A  
**Explanation**: `s > 3` gives [4, 5]. Sum = 4+5 = 9

---

## Q13. Output?
```python
import pandas as pd
df = pd.DataFrame({'A': [1, 2, None, 4]})
print(df['A'].isnull().sum())
```

**A)** `0`  
**B)** `1`  
**C)** `3`  
**D)** `4`

**Answer**: B  
**Explanation**: One `None` value = 1 null

---

## Q14. What prints?
```python
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3]})
df['B'] = df['A'] * 2
print(df['B'].iloc[1])
```

**A)** `2`  
**B)** `4`  
**C)** `6`  
**D)** `1`

**Answer**: B  
**Explanation**: `df['A'].iloc[1]` = 2, so `B` = 2*2 = 4

---

## Q15. Result?
```python
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.loc[1, 'B'])
```

**A)** `4`  
**B)** `5`  
**C)** `6`  
**D)** Error

**Answer**: B  
**Explanation**: `loc[1, 'B']` = row index 1, column 'B' = 5

---

## Q16. Output?
```python
import pandas as pd
df = pd.DataFrame({'A': [1, 1, 2, 2]})
print(df['A'].nunique())
```

**A)** `1`  
**B)** `2`  
**C)** `4`  
**D)** `0`

**Answer**: B  
**Explanation**: `nunique()` counts unique values: {1, 2} = 2 unique values

---

## Q17. What is the result?
```python
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3]})
df2 = df.copy()
df2.iloc[0, 0] = 99
print(df.iloc[0, 0])
```

**A)** `1`  
**B)** `99`  
**C)** Error  
**D)** `None`

**Answer**: A  
**Explanation**: `.copy()` creates independent copy. Changing df2 doesn't affect df.

---

## Q18. Output?
```python
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
print(df[df['A'] > 2].shape[0])
```

**A)** `2`  
**B)** `3`  
**C)** `5`  
**D)** `1`

**Answer**: B  
**Explanation**: Rows where A>2: [3,4,5] = 3 rows. `shape[0]` = number of rows = 3

---

## Q19. Result?
```python
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3]}, index=['x', 'y', 'z'])
print(df.loc['y', 'A'])
```

**A)** `1`  
**B)** `2`  
**C)** `3`  
**D)** Error

**Answer**: B  
**Explanation**: `loc['y', 'A']` uses label-based indexing: row 'y', column 'A' = 2

---

## Q20. Output?
```python
import pandas as pd
df1 = pd.DataFrame({'A': [1, 2]})
df2 = pd.DataFrame({'A': [3, 4]})
result = pd.concat([df1, df2])
print(result.shape)
```

**A)** `(2, 2)`  
**B)** `(4, 1)`  
**C)** `(2, 1)`  
**D)** `(1, 4)`

**Answer**: B  
**Explanation**: Vertical concatenation: 2 rows + 2 rows = 4 rows, 1 column → (4,1)

---

# Section 3: Scikit-Learn (10 Questions)

## Q21. What is the output?
```python
from sklearn.model_selection import train_test_split
X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
print(len(X_train))
```

**A)** `2`  
**B)** `4`  
**C)** `1`  
**D)** `3`

**Answer**: A  
**Explanation**: `test_size=0.5` means 50% test, 50% train. 4 samples → 2 train, 2 test

---

## Q22. Result?
```python
from sklearn.preprocessing import StandardScaler
import numpy as np
X = np.array([[1], [2], [3]])
scaler = StandardScaler()
scaler.fit(X)
print(scaler.mean_)
```

**A)** `[0]`  
**B)** `[1]`  
**C)** `[2]`  
**D)** `[3]`

**Answer**: C  
**Explanation**: Mean of [1, 2, 3] = (1+2+3)/3 = 2.0

---

## Q23. Output?
```python
from sklearn.linear_model import LinearRegression
import numpy as np
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])
model = LinearRegression()
model.fit(X, y)
print(model.coef_)
```

**A)** `[1]`  
**B)** `[2]`  
**C)** `[3]`  
**D)** `[0]`

**Answer**: B  
**Explanation**: Perfect linear relationship y=2x. Slope (coefficient) = 2

---

## Q24. What prints?
```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np
X = np.array([[1], [2], [3]])
y = np.array([0, 0, 1])
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)
print(clf.predict([[2]]))
```

**A)** `[0]`  
**B)** `[1]`  
**C)** `[2]`  
**D)** Error

**Answer**: A  
**Explanation**: Decision tree learns: X≤2.5 → class 0, X>2.5 → class 1. Input 2 → class 0

---

## Q25. Result?
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = ['cat', 'dog', 'cat', 'bird']
encoded = le.fit_transform(y)
print(encoded[0])
```

**A)** `0`  
**B)** `1`  
**C)** `2`  
**D)** `'cat'`

**Answer**: A  
**Explanation**: LabelEncoder assigns: 'bird'=0, 'cat'=1, 'dog'=2 (alphabetical). First 'cat' → 1. **CORRECTION**: Actually 'bird'=0, 'cat'=1, so 'cat'→1. But checking: typical alphabetical would give 'bird'=0, 'cat'=1, 'dog'=2, so first element 'cat'=1. **Wait**, let me recalculate: ['cat', 'dog', 'cat', 'bird'] → unique sorted: ['bird', 'cat', 'dog'] → mappings: bird=0, cat=1, dog=2. So 'cat'→1.

Actually, I need to be more careful. The correct answer should be **1** for 'cat'. Let me fix this.

**Answer**: B (should be 1)  
**Explanation**: Alphabetically: 'bird'=0, 'cat'=1, 'dog'=2. First element 'cat' → 1

---

## Q26. Output?
```python
from sklearn.metrics import accuracy_score
y_true = [1, 1, 0, 0]
y_pred = [1, 0, 0, 0]
print(accuracy_score(y_true, y_pred))
```

**A)** `0.5`  
**B)** `0.75`  
**C)** `1.0`  
**D)** `0.25`

**Answer**: B  
**Explanation**: Correct predictions: 3 out of 4 (indices 0, 2, 3). Accuracy = 3/4 = 0.75

---

## Q27. Result?
```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np
X = np.array([[1], [2], [3], [4]])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[0][0])
```

**A)** `0.0`  
**B)** `0.25`  
**C)** `0.5`  
**D)** `1.0`

**Answer**: A  
**Explanation**: MinMax: (x-min)/(max-min). For x=1: (1-1)/(4-1) = 0/3 = 0.0

---

## Q28. What is the output?
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])
model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=5)
print(len(scores))
```

**A)** `1`  
**B)** `5`  
**C)** `10`  
**D)** `25`

**Answer**: B  
**Explanation**: `cv=5` means 5-fold CV → returns array of 5 scores (one per fold)

---

## Q29. Result?
```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X, y)
print(len(rf.estimators_))
```

**A)** `1`  
**B)** `10`  
**C)** `3`  
**D)** `100`

**Answer**: B  
**Explanation**: `n_estimators=10` → 10 trees in forest. `estimators_` contains all 10 trees.

---

## Q30. Output?
```python
from sklearn.metrics import confusion_matrix
y_true = [1, 0, 1, 1]
y_pred = [1, 0, 0, 1]
cm = confusion_matrix(y_true, y_pred)
print(cm[0, 0])
```

**A)** `1`  
**B)** `2`  
**C)** `3`  
**D)** `0`

**Answer**: A  
**Explanation**: Confusion matrix:
```
         Pred 0  Pred 1
Actual 0    1      0
Actual 1    1      2
```
`cm[0,0]` = True Negatives = 1 (index 1: actual=0, pred=0)

---

# Answer Key

1-A, 2-A, 3-C, 4-A, 5-B, 6-B, 7-B, 8-A, 9-B, 10-C  
11-A, 12-A, 13-B, 14-B, 15-B, 16-B, 17-A, 18-B, 19-B, 20-B  
21-A, 22-C, 23-B, 24-A, 25-B, 26-B, 27-A, 28-B, 29-B, 30-A

---

# Common Traps to Avoid

1. **NumPy assignment is reference, not copy!** Use `.copy()`
2. **Pandas `loc` vs `iloc`**: loc=label-based, iloc=integer-based
3. **Axis confusion**: axis=0 → along rows (column-wise operation), axis=1 → along columns (row-wise operation)
4. **Boolean indexing**: Returns elements, not indices!
5. **Train/test split**: `test_size=0.2` means 80% train, 20% test
6. **Scaler must fit on train only**, then transform both train and test
7. **LabelEncoder** for target, **OneHotEncoder** for categorical features
8. **Cross_val_score** returns one score per fold
9. **Confusion matrix indexing**: `cm[actual, predicted]`

---

**End of Code Output Predictions**
