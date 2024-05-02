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
# data = data.drop(columns=['_id', 'student', 'flouciPaymentId'])


# Step 4: Preparing Data For modeling
print("Step 4: Preparing Data For Modeling")
X = data.drop(columns=['amount', 'method', 'status', 'type'])
y = data[['amount', 'method', 'status', 'type']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build Models
print("Step 5: Building Models")

# Numerical prediction for 'amount'
regressor_pipeline = Pipeline([
    ('regressor', RandomForestRegressor(random_state=42))
])

# Categorical predictions for 'method', 'status', 'type'
classifier_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ('classifier', RandomForestClassifier(random_state=42))
])


# Fit models
regressor_pipeline.fit(X_train, y_train['amount'])
classifier_pipeline.fit(X_train, y_train[['method', 'status', 'type']])

# Step 6: Evaluate Models
print("Step 6: Evaluating Models")
# Predictions
amount_pred = regressor_pipeline.predict(X_test)
method_status_type_pred = classifier_pipeline.predict(X_test)

# Evaluate RMSE for Amount
amount_rmse = mean_squared_error(y_test['amount'], amount_pred, squared=False)
print(f"RMSE for Amount: {amount_rmse}")

# Evaluate accuracy for Method
method_accuracy = accuracy_score(y_test['method'], method_status_type_pred[:, 0])
print(f"Accuracy for Method: {method_accuracy}")

# Evaluate accuracy for Status
status_accuracy = accuracy_score(y_test['status'], method_status_type_pred[:, 1])
print(f"Accuracy for Status: {status_accuracy}")

# Evaluate accuracy for Type
type_accuracy = accuracy_score(y_test['type'], method_status_type_pred[:, 2])
print(f"Accuracy for Type: {type_accuracy}")


# Step 7: Visualization
print("Step 7: Visualization")
plt.figure(figsize=(10, 5))
plt.plot(y_test['amount'].index, y_test['amount'], label='Actual Amount')
plt.plot(y_test['amount'].index, amount_pred, label='Predicted Amount', linestyle='--')
plt.legend()
plt.show()

