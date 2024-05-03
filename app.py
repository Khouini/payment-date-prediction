from flask import Flask, jsonify
import pandas as pd
from prophet import Prophet
import joblib

app = Flask(__name__)

# Load Prophet models
model_credits = joblib.load('model_credits.pkl')
model_debits = joblib.load('model_debits.pkl')

@app.route('/get_balance', methods=['GET'])
def get_balance():
    # Load historical data
    merged_data = pd.read_json("merged_data.json", convert_dates=['ds'])  # Ensure dates are converted
    
    # Predict future credits and debits
    future_credits = model_credits.make_future_dataframe(periods=12, freq='M')
    future_debits = model_debits.make_future_dataframe(periods=12, freq='M')
    
    forecast_credits = model_credits.predict(future_credits)
    forecast_debits = model_debits.predict(future_debits)
    
    # Calculate forecast balance
    forecast_credits['forecast_debits'] = forecast_debits['yhat']
    forecast_credits['predicted_balance'] = forecast_credits['yhat'] - forecast_credits['forecast_debits']
    
    # Calculate actual balance from historical data
    merged_data['actual_balance'] = merged_data['y_credit'] - merged_data['y_debit']
    
    # Ensure the 'ds' column is datetime and format it
    if merged_data['ds'].dtype != 'datetime64[ns]':
        merged_data['ds'] = pd.to_datetime(merged_data['ds'])
    merged_data['month'] = merged_data['ds'].dt.strftime('%m:%Y')
    forecast_credits['month'] = forecast_credits['ds'].dt.strftime('%m:%Y')
    
    # Convert to JSON with modified month format
    actual_balance_json = merged_data[['month', 'actual_balance']].to_json(orient='records')
    predicted_balance_json = forecast_credits[['month', 'predicted_balance']].to_json(orient='records')
    
    return jsonify(actual_balance=actual_balance_json, predicted_balance=predicted_balance_json)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
