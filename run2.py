import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 2: Load and Prepare the Data
data = pd.read_csv('fake_payments.csv')
data['createdAt'] = pd.to_datetime(data['createdAt'])
data.set_index('createdAt', inplace=True)

# Step 3: Feature Engineering
data['day_of_week'] = data.index.dayofweek
data['hour_of_day'] = data.index.hour
data['day_of_month'] = data.index.day
data['week_of_year'] = data.index.isocalendar().week

# Step 4: Preparing Data For Modeling
X = data.drop(columns=['amount', 'method', 'status', 'type'])
y = data[['amount', 'method', 'status', 'type']]

numeric_features = ['day_of_month', 'week_of_year']
categorical_features = ['day_of_week', 'hour_of_day']

# Preprocessing
numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Separate target features
y_amount = y['amount']
y_category = y[['method', 'status', 'type']]

X_train, X_test, y_train_amount, y_test_amount = train_test_split(X, y_amount, test_size=0.2, random_state=42)
_, _, y_train_category, y_test_category = train_test_split(X, y_category, test_size=0.2, random_state=42)

# Step 5: Build Models
# Numerical prediction for 'amount'
regressor_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Categorical predictions for 'method', 'status', 'type'
classifier_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit models
regressor_pipeline.fit(X_train, y_train_amount)
classifier_pipeline.fit(X_train, y_train_category)

# Step 6: Evaluate Models
from sklearn.metrics import classification_report

# Predictions
amount_pred = regressor_pipeline.predict(X_test)
method_pred, status_pred, type_pred = classifier_pipeline.predict(X_test).T  # Transpose to separate predictions

# Evaluate RMSE for Amount
amount_rmse = mean_squared_error(y_test_amount, amount_pred, squared=False)
print(f"RMSE for Amount: {amount_rmse}")

# Classification reports for each category
print("Classification Report for Method:")
print(classification_report(y_test_category['method'], method_pred))

print("Classification Report for Status:")
print(classification_report(y_test_category['status'], status_pred))

print("Classification Report for Type:")
print(classification_report(y_test_category['type'], type_pred))
