import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import json

# Load data from JSON
with open("fake_payments.json", "r") as f:
    payments = json.load(f)

# Create DataFrame
df = pd.DataFrame(payments)

# Convert 'createdAt' to datetime
df['createdAt'] = pd.to_datetime(df['createdAt'])

# Resample and sum credit and debit amounts by month
monthly_credits = df[df['type'] == 'credit'].resample('M', on='createdAt').amount.sum().reset_index()
monthly_debits = df[df['type'] == 'debit'].resample('M', on='createdAt').amount.sum().reset_index()

# Rename columns for Prophet compatibility
monthly_credits.rename(columns={'createdAt': 'ds', 'amount': 'y_credit'}, inplace=True)
monthly_debits.rename(columns={'createdAt': 'ds', 'amount': 'y_debit'}, inplace=True)

# Merge the data on date, handling missing months
merged_data = pd.merge(monthly_credits, monthly_debits, on='ds', how='outer')
merged_data.fillna(0, inplace=True)  # Assuming no transactions occurred in missing months

# Prepare datasets for Prophet
credits_data = merged_data[['ds', 'y_credit']].rename(columns={'y_credit': 'y'})
debits_data = merged_data[['ds', 'y_debit']].rename(columns={'y_debit': 'y'})

# Initialize Prophet models for credit and debit predictions
model_credits = Prophet(yearly_seasonality=True)
model_debits = Prophet(yearly_seasonality=True)

# Fit the models
model_credits.fit(credits_data)
model_debits.fit(debits_data)

# Make future dataframes for forecasting
future_credits = model_credits.make_future_dataframe(periods=12, freq='M')
future_debits = model_debits.make_future_dataframe(periods=12, freq='M')

# Predict future credits and debits
forecast_credits = model_credits.predict(future_credits)
forecast_debits = model_debits.predict(future_debits)

# Calculate forecast balance
forecast_credits['forecast_debits'] = forecast_debits['yhat']
forecast_credits['predicted_balance'] = forecast_credits['yhat'] - forecast_credits['forecast_debits']

# Calculate actual balance from historical data
merged_data['actual_balance'] = merged_data['y_credit'] - merged_data['y_debit']

print(merged_data['actual_balance'])
print(forecast_credits['predicted_balance'])
# return merged_data['actual_balance'], forecast_credits['predicted_balance'] as json with flask, dont forget saving the model

import joblib

# Save Prophet models
joblib.dump(model_credits, 'model_credits.pkl')
joblib.dump(model_debits, 'model_debits.pkl')

# Load Prophet models
loaded_model_credits = joblib.load('model_credits.pkl')
loaded_model_debits = joblib.load('model_debits.pkl')

# Save merged_data as JSON
merged_data.to_json('merged_data.json', orient='records')
