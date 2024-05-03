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

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot actual balance
plt.plot(merged_data['ds'], merged_data['actual_balance'], label='Actual Balance', color='red')
print(merged_data['actual_balance'])
# Plot forecasted balance
plt.plot(forecast_credits['ds'], forecast_credits['predicted_balance'], label='Forecasted Balance', color='blue')
print(forecast_credits['predicted_balance'])

plt.title('Actual vs. Forecasted Balance')
plt.xlabel('Date')
plt.ylabel('Balance')
plt.legend()
plt.grid(True)  # Add grid for better readability
plt.show()

from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

aligned_forecast = forecast_credits.set_index('ds').loc[merged_data['ds']].reset_index()

# Calculate R-squared
r_squared = r2_score(merged_data['actual_balance'], aligned_forecast['predicted_balance'])

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(merged_data['actual_balance'], aligned_forecast['predicted_balance'])

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

print("R-squared:", r_squared)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)


# Calculate R-squared
# R-squared (R²) is like a report card for your model's performance. It tells you how well the predictions match the actual data. R-squared ranges from 0 to 1, where 1 means the model perfectly predicts the data and 0 means it's no better than guessing. So, the closer R-squared is to 1, the better your model fits the data.

# Calculate Mean Squared Error (MSE)
# Mean Squared Error (MSE) is like measuring the average size of mistakes your model makes. It takes the difference between each prediction and the actual value, squares it to get rid of negative values, adds up all those squared differences, and then divides by the number of predictions. A smaller MSE means your model's predictions are closer to the actual values on average.

# Calculate Root Mean Squared Error (RMSE)
# Root Mean Squared Error (RMSE) is like the "easy-to-read" version of MSE. It's the square root of the MSE, so it gives you the average size of the errors in the same units as the original data. For example, if you're predicting prices in dollars, RMSE will tell you, on average, how many dollars your predictions are off by. Lower RMSE values indicate better performance.

# R-squared: 0.9999175079851397
# Mean Squared Error (MSE): 93044.63458749463
# Root Mean Squared Error (RMSE): 305.03218615007603
# which is great which not

# With an R-squared value very close to 1 (0.9999), it suggests that the model explains almost all of the variance in the actual balance. This indicates an excellent fit between the predicted and actual values, meaning the model captures the underlying patterns in the data very well.

# Regarding the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE), both values are relatively low. The MSE value of 93044.63 suggests that, on average, the squared difference between the predicted and actual balances is around 93044.63. The RMSE value of 305.03, being the square root of MSE, indicates that, on average, the model's predictions are off by approximately 305 units of the same scale as the actual balance.

# In summary, based on these evaluation metrics, the model performs exceptionally well in predicting the balance. However, it's always recommended to consider the context of the problem domain and compare these values against other models or benchmarks to get a comprehensive understanding of the model's performance.
