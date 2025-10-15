# Scikit-Learn Guide

Quick reference for scikit-learn, the go-to library for classical machine learning.

## Installation
```bash
pip install scikit-learn
```

## Quick Examples

### Classification
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
```

### Regression
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
```

### Clustering
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X)
```

## Common Algorithms

### Supervised Learning
- `LinearRegression`, `Ridge`, `Lasso`
- `LogisticRegression`
- `DecisionTreeClassifier/Regressor`
- `RandomForestClassifier/Regressor`
- `GradientBoostingClassifier/Regressor`
- `SVC`, `SVR` (Support Vector Machines)
- `KNeighborsClassifier/Regressor`

### Unsupervised Learning
- `KMeans`, `DBSCAN` (Clustering)
- `PCA` (Dimensionality Reduction)
- `IsolationForest` (Anomaly Detection)

## Preprocessing

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Encoding
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
```

## Model Selection

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"CV Score: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Hyperparameter tuning
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

## Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Cheat Sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
