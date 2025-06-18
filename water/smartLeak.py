import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Starting Smart Water Meter Leak Detection Script ---")

# --- Step 1: Data Loading ---

file_path = 'smart.txt' # Make sure this file is in the same directory

try:
    df = pd.read_csv(file_path, parse_dates=['Timestamp']) # parse_dates helps treat Timestamp as proper datetime
    print(f"Step 1: Data loaded successfully from '{file_path}'.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please ensure the 'smart.txt' file is in the same directory as the script.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    exit()

# Optional: Display first few rows and info to check data
# print("\n--- First 5 Rows of the Data ---")
# print(df.head())
# print("\n--- Data Information (df.info()) ---")
# df.info()

# --- Step 1.5: Robust Data Cleaning for 'Is_Leak' column ---
# This is the CRUCIAL part to ensure 'Is_Leak' only contains 0 or 1 as integers.

# First, replace the specific problematic string with '1'
problematic_string = '1 <-- بداية التسريب في الليل (استهلاك أعلى من المعتاد)'
if problematic_string in df['Is_Leak'].astype(str).unique(): # Check if it exists before replacing
    df['Is_Leak'] = df['Is_Leak'].replace(problematic_string, '1')

# Now, convert the entire 'Is_Leak' column to numeric, coercing any remaining errors to NaN
# Then, fill any NaNs (if any appeared due to other non-numeric entries) and convert to integer
df['Is_Leak'] = pd.to_numeric(df['Is_Leak'], errors='coerce')
df['Is_Leak'] = df['Is_Leak'].fillna(0).astype(int) # Fill any potential NaNs (e.g., from other bad strings) with 0 and convert to int

# Verify the unique values and their types after cleaning
print("\n--- Unique values in 'Is_Leak' after cleaning ---")
print(df['Is_Leak'].unique())
print(f"Data type of 'Is_Leak' after cleaning: {df['Is_Leak'].dtype}")


# --- Step 2: Data Preprocessing and Feature Selection ---

target_variable = 'Is_Leak'
feature_variables = ['Consumption_m3']

X = df[feature_variables] # Input features
Y = df[target_variable]   # Target variable (NOW CLEANED)

print(f"\nStep 2: Defined Target Variable (Y): {target_variable}")
print(f"Step 2: Defined Feature Variables (X): {feature_variables}")

# Split Data into Training and Testing Sets
# Keep stratify=Y if you have enough samples in the minority class (which you should after cleaning)
# If it still fails with stratify=Y, remove it.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

print(f"Step 2: Data split into Training ({X_train.shape[0]} samples) and Testing ({X_test.shape[0]} samples) sets.")
print(f"Training set leak distribution:\n{Y_train.value_counts(normalize=True)}")
print(f"Testing set leak distribution:\n{Y_test.value_counts(normalize=True)}")


# --- Step 3: Model Building, Training, and Evaluation ---

# 1. Choose and Initialize the Model (Logistic Regression for binary classification)
model = LogisticRegression(random_state=42)
print("\nStep 3: Logistic Regression model initialized.")

# 2. Train the Model
model.fit(X_train, Y_train)
print("Step 3: Model training complete.")

# 3. Make Predictions on the Test Set
Y_pred = model.predict(X_test)
print("Step 3: Predictions made on the test set.")

# 4. Evaluate the Model's Performance
# The zero_division parameter is crucial for cases where a class has no true predicted values
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, zero_division=0)
recall = recall_score(Y_test, Y_pred, zero_division=0)
f1 = f1_score(Y_test, Y_pred, zero_division=0)
conf_matrix = confusion_matrix(Y_test, Y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("  (True Negative | False Positive)")
print("  (False Negative | True Positive)")


# 5. Visualize Predictions vs Actual (Optional for Classification, but useful)
plt.figure(figsize=(10, 6))
# Filter df for plotting to only include columns needed
plot_df = df.copy() # Make a copy to avoid SettingWithCopyWarning
plot_df = plot_df[['Timestamp', 'Consumption_m3', 'Is_Leak']] # Select only relevant columns
plot_df['Is_Pred_Leak'] = pd.Series(model.predict(X), index=df.index) # Predict on ALL data for full plot range

# Plot overall consumption trend
plt.plot(plot_df['Timestamp'], plot_df['Consumption_m3'], color='grey', linestyle='--', alpha=0.5, label='Overall Consumption Trend')

# Plot actual leak points
plt.scatter(plot_df.loc[plot_df['Is_Leak'] == 1, 'Timestamp'], plot_df.loc[plot_df['Is_Leak'] == 1, 'Consumption_m3'],
            color='red', marker='x', s=100, label='Actual Leak') # Removed (Test Set) as we plot all actual leaks

# Plot predicted leak points (only where predicted as 1)
plt.scatter(plot_df.loc[plot_df['Is_Pred_Leak'] == 1, 'Timestamp'], plot_df.loc[plot_df['Is_Pred_Leak'] == 1, 'Consumption_m3'],
            color='blue', marker='o', alpha=0.5, label='Predicted Leak')


plt.title('Actual vs. Predicted Leakage Based on Consumption')
plt.xlabel('Timestamp')
plt.ylabel('Consumption (m³)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('leak_detection_predictions.png')
print("\nStep 3: Visualization saved as 'leak_detection_predictions.png'")


print("\n--- Script Execution Complete ---")
print("This model demonstrates how consumption data can be used to classify potential leaks.")