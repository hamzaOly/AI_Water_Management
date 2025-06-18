# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 21:05:00 2025

@author: YourName (Hamza Eleimat)

Description:
Consolidated Python script for building a simple AI model to predict Annual Water Deficit
in Jordan using historical data.
Includes data loading, preprocessing, model training, and evaluation.
"""

import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

print("--- Starting Water Deficit Prediction Model Script ---")

# --- Step 1: Data Loading and Initial Inspection ---

# IMPORTANT:
# Replace the StringIO part below with the actual path to your CSV file when running on your machine.
# Example: df = pd.read_csv('your_actual_dataset_name.csv')
# Or for Excel: df = pd.read_excel('your_actual_dataset_name.xlsx')

file_path = 'water_data.csv' # تأكد أن هذا هو الاسم الصحيح لملفك

try:
    # جرب هذا التعديل: إضافة encoding='utf-8'
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"Step 1: Data loaded successfully from '{file_path}' with UTF-8 encoding.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please ensure the file is in the same directory as the script, or provide the full path.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    
    try:
        df = pd.read_csv(file_path, encoding='latin1')
        print(f"Step 1: Data loaded successfully from '{file_path}' with latin1 encoding.")
    except Exception as e2:
        print(f"Failed to load with latin1 either: {e2}")
        exit()

# After loading the data, run the columns again to make sure.
print("\n--- Actual Columns in the Loaded Data (after stripping spaces) ---")
df.columns = df.columns.str.strip() # Spaces
print(df.columns)
print("-------------------------------------------\n")
# --- Step 2: Data Preprocessing and Feature Selection ---

# Define Target Variable (Y) and Feature Variables (X)
target_variable = 'Annual Water Deficit (billion m³)'
feature_variables = [
    'Annual Rainfall (mm)',
    'Freshwater Withdrawal (billion m³)',
    'Water Stress (%)'
]

X = df[feature_variables] # Input features
Y = df[target_variable]   # Target variable

print(f"\nStep 2: Defined Target Variable (Y): {target_variable}")
print(f"Step 2: Defined Feature Variables (X): {feature_variables}")


# Handle Missing Values (if any - based on previous check, these columns are clean for this dataset)
# If your real data has missing values in these specific columns, uncomment and adjust the following:
# X = X.fillna(X.mean()) # Example: fill missing X values with their column mean
# Y = Y.fillna(Y.mean()) # Example: fill missing Y values with its mean


# Split Data into Training and Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Step 2: Data split into Training ({X_train.shape[0]} samples) and Testing ({X_test.shape[0]} samples) sets.")


# --- Step 3: Model Building, Training, and Evaluation ---

# 1. Choose and Initialize the Model
model = LinearRegression()
print("\nStep 3: Linear Regression model initialized.")

# 2. Train the Model
model.fit(X_train, Y_train)
print("Step 3: Model training complete.")

# Display the coefficients and intercept
print("\n--- Model Coefficients and Intercept ---")
for feature, coef in zip(feature_variables, model.coef_):
    print(f"Coefficient for '{feature}': {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# 3. Make Predictions on the Test Set
Y_pred = model.predict(X_test)
print("Step 3: Predictions made on the test set.")

# 4. Evaluate the Model's Performance
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2 Score): {r2:.4f}")


# 5. Visualize Actual vs. Predicted Values (for the Test Set)
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, color='blue', label='Actual vs. Predicted')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--', label='Ideal Prediction Line')
plt.title('Actual vs. Predicted Annual Water Deficit')
plt.xlabel('Actual Annual Water Deficit (billion m³)')
plt.ylabel('Predicted Annual Water Deficit (billion m³)')
plt.legend()
plt.grid(True)
plt.savefig('actual_vs_predicted_water_deficit.png') # Save the plot
print("\nStep 3: Visualization saved as 'actual_vs_predicted_water_deficit.png'")

print("\n--- Script Execution Complete ---")
print("You now have a trained model and its evaluation metrics. Use these in your presentation!")