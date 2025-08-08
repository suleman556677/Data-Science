import numpy as np
import matplotlib as plt
import pandas as pd
import seaborn as sns

# Load the diabetes dataset from sklearn

from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
diabetes.target[:3]
diabetes.data.shape

# Load the diabetes dataset

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
print("Data shape:", X.shape)
print("Target shape:", y.shape)
print("First 3 target values:", y[:3])

# Convert to DataFrame for better visualization

df = pd.DataFrame(X, columns=diabetes.feature_names)
df['target'] = y
print(df.head())
print(df.describe())

# Visualize the distribution of the target variable

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a linear regression model

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics

print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Visualize the predictions

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted - Diabetes Dataset")
plt.show()
print("wait here")
