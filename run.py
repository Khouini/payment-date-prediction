# Step 1: Import Necessary Libraries
print("Step 1: Importing Necessary Libraries")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

# Step 2: Load and Prepare the Data
print("Step 2: Loading and Preparing the Data")
# Load your data
data = pd.read_csv('fake_payments.csv')
data['createdAt'] = pd.to_datetime(data['createdAt'])
data.set_index('createdAt', inplace=True)

print(data.head())

# Step 3: Feature Engineering
print("Step 3: Feature Engineering")

# Add time-based features
data['day_of_week'] = data.index.dayofweek
data['hour_of_day'] = data.index.hour

# Optionally, drop unnecessary or redundant features
# data = data.drop(columns=['_id', 'student', 'flouciPaymentId', 'method', 'status'])
data = data.drop(columns=['method', 'status'])

# Step 4: Preparing Data For Modeling
print("Step 4: Preparing Data For Modeling")
X = data.drop(columns=['amount', 'type'])
y = data[['amount', 'type']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build Models
print("Step 5: Building Models")

# Numerical prediction for 'amount'
amount_regressor = Pipeline([
    ('regressor', RandomForestRegressor(random_state=42))
])

# Categorical prediction for 'type'
type_classifier = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit models
amount_regressor.fit(X_train, y_train['amount'])
type_classifier.fit(X_train, y_train['type'])

# Step 6: Evaluate Models
print("Step 6: Evaluating Models")
# Predictions
amount_pred = amount_regressor.predict(X_test)
type_pred = type_classifier.predict(X_test)

# Evaluate RMSE for Amount
amount_rmse = mean_squared_error(y_test['amount'], amount_pred, squared=False)
print(f"RMSE for Amount: {amount_rmse}")

# Evaluate accuracy for Type
type_accuracy = accuracy_score(y_test['type'], type_pred)
print(f"Accuracy for Type: {type_accuracy}")

# Step 7: Visualization

# Adjust 'y_test' amounts based on type
y_test_adjusted = y_test.copy()
y_test_adjusted['adjusted_amount'] = np.where(y_test_adjusted['type'] == 'credit', y_test_adjusted['amount'], -y_test_adjusted['amount'])

# Adjust 'amount_pred' based on predicted types
amount_pred_adjusted = np.where(type_pred == 'credit', amount_pred, -amount_pred)

# Convert 'amount_pred_adjusted' from a numpy array to a pandas Series with the same datetime index as 'y_test['amount']'
amount_pred_series_adjusted = pd.Series(amount_pred_adjusted, index=y_test['amount'].index)

# Now we can resample because 'amount_pred_series_adjusted' is a pandas Series with a datetime index
monthly_predicted_adjusted = amount_pred_series_adjusted.resample('M').sum()

# Group the actual adjusted amounts by month
monthly_actual_adjusted = y_test_adjusted['adjusted_amount'].resample('M').sum()

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(monthly_actual_adjusted.index, monthly_actual_adjusted, label='Actual Adjusted Amount')
plt.plot(monthly_predicted_adjusted.index, monthly_predicted_adjusted, label='Predicted Adjusted Amount', linestyle='--')
plt.title('Monthly Actual vs Predicted Adjusted Amounts')
plt.xlabel('Month')
plt.ylabel('Adjusted Amount')
plt.legend()
plt.show()
