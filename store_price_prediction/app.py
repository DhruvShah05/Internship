from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
import joblib
from datetime import timedelta

# Set Matplotlib backend to 'Agg' for running in non-interactive mode
plt.switch_backend('Agg')

app = Flask(__name__)

def load_store_data_and_model(store_nbr):
    data = pd.read_csv(f'store_datasets/store_{store_nbr}_data.csv')
    model = load_model(f'store_models/store_{store_nbr}_model.keras')
    scaler = joblib.load(f'store_models/store_{store_nbr}_scaler.pkl')
    return data, model, scaler

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_store', methods=['GET', 'POST'])
def select_store():
    if request.method == 'POST':
        store_number = int(request.form['store-number'])
        return redirect(url_for('select_time_period', store_number=store_number))
    return render_template('select_store.html')

@app.route('/select_time_period/<int:store_number>', methods=['GET', 'POST'])
def select_time_period(store_number):
    if request.method == 'POST':
        time_period = request.form['time-period']
        return redirect(url_for('visualizations', store_number=store_number, time_period=time_period))
    return render_template('select_time_period.html', store_number=store_number)

@app.route('/visualizations/<int:store_number>/<time_period>')
def visualizations(store_number, time_period):
    # Load store data, model, and scaler
    df, model, scaler = load_store_data_and_model(store_number)
    
    # Ensure the date column is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Prepare the data
    data = df['daily_sales'].values.reshape(-1, 1)
    scaled_data = scaler.transform(data)
    
    # Split the data
    train_data = scaled_data[:-10]
    test_data = scaled_data[-10:]
    
    # Create sequences for prediction
    seq_length = 30
    X_test = np.array([train_data[-seq_length:]])
    
    # Generate predictions for the test set
    test_predictions = []
    for _ in range(10):
        next_pred = model.predict(X_test)
        test_predictions.append(next_pred[0, 0])
        X_test = np.roll(X_test, -1, axis=1)
        X_test[0, -1, 0] = next_pred[0, 0]
    
    # Inverse transform the predictions
    test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten()
    
    # Plot 1: Actual vs Predicted (last 10 days)
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'].tail(10), df['daily_sales'].tail(10), label='Actual', marker='o')
    plt.plot(df['date'].tail(10), test_predictions, label='Predicted', marker='x')
    plt.title(f'Actual vs Predicted Daily Sales for Store {store_number} (Last 10 Days)')
    plt.xlabel('Date')
    plt.ylabel('Daily Sales')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join('static', 'plot', f'store_{store_number}_actual_vs_predicted.png')
    plt.savefig(plot_path)
    plt.close()

    # User-specified forecast period
    forecast_periods = {'1-day': 1, '1-week': 7, '1-month': 30, '2-months': 60, '3-months': 90, 
                        '4-months': 120, '5-months': 150, '6-months': 180, '7-months': 210, 
                        '8-months': 240, '9-months': 270, '10-months': 300, '11-months': 330, 
                        '1-year': 365}
    user_forecast_steps = forecast_periods[time_period]
    
    # Generate future forecast
    X_forecast = np.array([scaled_data[-seq_length:]])
    forecast = []
    for _ in range(user_forecast_steps):
        next_pred = model.predict(X_forecast)
        forecast.append(next_pred[0, 0])
        X_forecast = np.roll(X_forecast, -1, axis=1)
        X_forecast[0, -1, 0] = next_pred[0, 0]
    
    # Inverse transform the forecast
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    
    # Generate future dates for the forecast
    last_date = df['date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(user_forecast_steps)]
    
    # Plot 2: Full dataset with future forecast
    plt.figure(figsize=(14, 7))
    plt.plot(df['date'], df['daily_sales'], label='Historical Daily Sales')
    plt.plot(future_dates, forecast, label='Future Forecast', color='red')
    plt.xlabel('Date')
    plt.ylabel('Daily Sales')
    plt.title(f'Historical and Forecasted Daily Sales for Store {store_number}')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    full_plot_path = os.path.join('static', 'plot', f'store_{store_number}_full_forecast.png')
    plt.savefig(full_plot_path)
    plt.close()

    # Plot 3: Last 90 days of actual data + future forecast
    last_90_days = df.tail(90)
    plt.figure(figsize=(14, 7))
    plt.plot(last_90_days['date'], last_90_days['daily_sales'], label='Actual (Last 90 Days)', color='blue')
    plt.plot(future_dates, forecast, label='Future Forecast', color='red')
    plt.xlabel('Date')
    plt.ylabel('Daily Sales')
    plt.title(f'Last 90 Days and Future Forecast for Store {store_number}')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    combined_plot_path = os.path.join('static', 'plot', f'store_{store_number}_combined_forecast.png')
    plt.savefig(combined_plot_path)
    plt.close()

    return render_template('visualizations.html', store_number=store_number, plot_path=plot_path, full_plot_path=full_plot_path, combined_plot_path=combined_plot_path)
if __name__ == '__main__':
    app.run(debug=True)