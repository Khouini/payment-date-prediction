import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Load data from CSV
file_path = 'fake_payments.csv'  # Update this path to your CSV file location
data = pd.read_csv(file_path)
data['createdAt'] = pd.to_datetime(data['createdAt'])
data.set_index('createdAt', inplace=True)

# Calculate net payments per month
monthly_data = data.groupby([data.index.year, data.index.month, 'type']).sum()['amount'].unstack(fill_value=0)
monthly_data['net'] = monthly_data.get('credit', 0) - monthly_data.get('debit', 0)

# Create a proper datetime index
monthly_data.index = pd.to_datetime(['{}-{}'.format(*idx) for idx in monthly_data.index])

# Check for stationarity
result = adfuller(monthly_data['net'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# Fit ARIMA model using auto_arima to find best parameters
try:
    model = auto_arima(monthly_data['net'], seasonal=False, trace=True,  # Disable seasonal differencing
                       error_action='ignore', suppress_warnings=True, stepwise=True)
    model.fit(monthly_data['net'])

    # Forecast next 12 months
    forecast = model.predict(n_periods=12)
    forecast_index = pd.date_range(start=monthly_data.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_data.index, monthly_data['net'], label='Historical Net Payments')
    plt.plot(forecast_index, forecast, color='red', label='Forecasted Net Payments')
    plt.title('Net Payment Forecast')
    plt.xlabel('Date')
    plt.ylabel('Net Payment Amount')
    plt.legend()
    plt.show()
except ValueError as e:
    print("Error encountered: ", e)
    print("Consider reviewing the series length, seasonality settings, or model complexity.")
