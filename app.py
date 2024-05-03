from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load your machine learning models
with open('amount_regressor.pkl', 'rb') as f:
    loaded_amount_regressor = pickle.load(f)
with open('type_classifier.pkl', 'rb') as f:
    loaded_type_classifier = pickle.load(f)

@app.route('/predict', methods=['GET'])
def predict():
    # Extract date range from query parameters
    start_date = request.args.get('start_date', type=pd.Timestamp)
    end_date = request.args.get('end_date', type=pd.Timestamp)

    df = pd.read_csv('fake_payments.csv')
    # Generate or load your data here according to the date range
    # For demonstration, we'll assume you have a DataFrame 'df' loaded and filtered by date range
    df_filtered = df[(df['createdAt'] >= start_date) & (df['createdAt'] <= end_date)]

    # Perform predictions
    amount_pred = loaded_amount_regressor.predict(df_filtered)
    type_pred = loaded_type_classifier.predict(df_filtered)

    # Adjust predictions based on type
    amount_pred_adjusted = np.where(type_pred == 'credit', amount_pred, -amount_pred)

    # Convert predictions to pandas Series and group by month
    amount_pred_series_adjusted = pd.Series(amount_pred_adjusted, index=df_filtered['createdAt'].index)
    monthly_predicted_adjusted = amount_pred_series_adjusted.resample('M').sum()

    # Group the actual amounts by month and adjust
    y_test_adjusted = df_filtered.copy()
    y_test_adjusted['adjusted_amount'] = np.where(y_test_adjusted['type'] == 'credit', y_test_adjusted['amount'], -y_test_adjusted['amount'])
    monthly_actual_adjusted = y_test_adjusted['adjusted_amount'].resample('M').sum()

    # Create the response dictionary
    response_dict = {}
    for date in monthly_actual_adjusted.index:
        month_str = date.strftime('%Y-%m')
        response_dict[month_str] = {
            "actual_total": monthly_actual_adjusted[date],
            "predicted_total": monthly_predicted_adjusted[date],
            "details": {
                "actual_payments": y_test_adjusted['amount'][y_test_adjusted.index.month == date.month].tolist(),
                "predicted_payments": amount_pred_series_adjusted[amount_pred_series_adjusted.index.month == date.month].tolist()
            }
        }

    return jsonify(response_dict)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

