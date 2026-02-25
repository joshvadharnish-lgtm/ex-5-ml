# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, preprocess the data, apply one-hot encoding, and standardize the features and target variable.

2.Split the dataset into training and testing sets.

3.Train Ridge, Lasso, and ElasticNet models using polynomial features and make predictions on test data.

4.Calculate MSE and R² score for each model and compare their performance using bar charts. 

## Program:
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv("encoded_car_data (1) (1).csv")
print(data.head())

data = pd.get_dummies(data, drop_first=True)X = data.drop('price', axis=1)
y = data['price']
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
}results = {}for name, model in models.items():
    # Create a pipeline with polynomial features and the model
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('regressor', model)
    ]) pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
  mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    results[name] = {'MSE': mse, 'R2 Score': r2}
print('=' * 50)
for model_name, metrics in results.items():
    print(f"{model_name} - Mean Squared Error: {metrics['MSE']:.2f}, R² Score: {metrics['R2 Score']:.2f}")
print('=' * 50)
results_df = pd.DataFrame(results).T.reset_index()
results_df.rename(columns={'index': 'Model'}, inplace=True)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(x="Model", y="MSE", data=results_df, palette="viridis")
plt.title("Mean Squared Error (MSE)")
plt.ylabel("MSE")
plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
sns.barplot(x="Model", y="R2 Score", data=results_df, palette="viridis")
plt.title("R² Score")
plt.ylabel("R² Score")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

## Output:
<img width="735" height="568" alt="image" src="https://github.com/user-attachments/assets/9339d3b5-2b72-48b9-92ea-5b53d8c876ea" />
![Uploading Screenshot 2026-02-25 093143.png…]()



## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
