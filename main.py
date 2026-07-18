import os
import numpy as np
import pandas as pd
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Download the latest version of the House Price Prediction dataset
path = kagglehub.dataset_download(
    "zafarali27/house-price-prediction-dataset"
)

print(f"Dataset downloaded to: {path}")

# Find the CSV file
dataset_files = os.listdir(path)
csv_files = [file for file in dataset_files if file.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError("No CSV file found in the dataset directory.")

house_data_path = os.path.join(path, csv_files[0])

print(f"Loading dataset from: {house_data_path}")

# Load dataset
house_df = pd.read_csv(house_data_path)

print("\nOriginal Dataset Columns:")
print(house_df.columns.tolist())

# Remove unnecessary columns if they exist
columns_to_drop = ["id", "date", "Unnamed: 0"]

house_df = house_df.drop(
    columns=[col for col in columns_to_drop if col in house_df.columns],
    errors="ignore",
)

# Define target variable
if "Price" not in house_df.columns:
    raise ValueError("Target column 'Price' not found in the dataset.")

# Features and target
X = house_df.drop("Price", axis=1)
y = house_df["Price"]

# Remove non-numeric columns
non_numeric_cols = X.select_dtypes(include=["object"]).columns

if len(non_numeric_cols) > 0:
    print("\nDropping non-numeric columns:")
    print(non_numeric_cols.tolist())
    X = X.drop(columns=non_numeric_cols)

# Fill missing values
X = X.fillna(X.mean())

print(f"\nFeature Matrix Shape: {X.shape}")
print(X.head())

print(f"\nTarget Shape: {y.shape}")
print(y.head())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print("\nTraining Linear Regression Model...")

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nMaking Predictions...")

# Predict
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("-" * 40)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")
