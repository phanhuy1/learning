# Ghi chú

## Lesson 1
feature = predictor variable = independent variable
target variable = dependent variable = response variable

Trước khi sử dụng supervised learning cần có các yêu cầu sau:
* No missing values
* Data in numeric format
* Data stored inn pandas dataframe or numpy array
Supervised Learning with scikit-learnSupervised Learning with scikit-learn
Thực hiện EDA first

scikit-learn syntax
```python
from sklearn.module import Model
model = Model()
model.fit(X, y)
predictions = model.predict(X_new)
print(predictions)

```
### Practice:
Binary classification
There are two types of supervised learning—classification and regression. Binary classification is used to predict a target variable that has only two labels, typically represented numerically with a zero or a one.

The `.head()` of a dataset, `churn_df`, is shown below. You can expect the rest of the data to contain similar values.
| Index | account_length | total_day_charge | total_eve_charge | total_night_charge | total_intl_charge | customer_service_calls | churn |
|-------|----------------|------------------|------------------|--------------------|-------------------|------------------------|-------|
| 0     | 101            | 45.85            | 17.65            | 9.64               | 1.22              | 3                      | 1     |
| 1     | 73             | 22.30            | 9.05             | 9.98               | 2.75              | 2                      | 0     |
| 2     | 86             | 24.62            | 17.53            | 11.49              | 3.13              | 4                      | 0     |
| 3     | 59             | 34.73            | 21.02            | 9.66               | 3.24              | 1                      | 0     |
| 4     | 129            | 27.42            | 18.75            | 10.11              | 2.59              | 1                      | 0     |

Looking at this data, which column could be the target variable for binary classification?

## Using scikit-learn to fit a classifier KNN
```python
from sklearn.neighbors import  KNeighbors
```

## Measuring Model Performance

ACCURACY
How do we measure accuracy
 
$\frac{correct predictions}{total observations}$

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
```

| Tham số    | Ý nghĩa                                                                   |
| ---------- | ------------------------------------------------------------------------- |
| `stratify` | Đảm bảo tỷ lệ phân bố của các lớp (nhãn) được giữ nguyên khi chia dữ liệu |


