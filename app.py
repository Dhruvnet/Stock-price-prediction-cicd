import os
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request

# Create Flask app
app = Flask(__name__)

# Path to your pickle model file
model_path = r'C:/Users/Dhruv/Desktop/ADS/ADS_Exp9/model/improved_rf_model.pkl'

# Load the trained model and scaler
with open(model_path, 'rb') as file:
    model, scaler = pickle.load(file)

# Define companies for the form and prediction targets
companies = ['BHARTIARTL.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'SBIN.NS', 'TCS.NS']

@app.route('/')
def index():
    return render_template('index.html', companies=companies)

@app.route('/predict', methods=['POST'])
def predict():
    # Fetch data for all companies
    data = []
    for company in companies:
        try:
            company_data = request.form[company]
            data.append(float(company_data))
        except ValueError:
            return "Please enter valid numeric values."

    # Reshape and scale the input data
    input_data = np.array([data])
    scaled_data = scaler.transform(input_data)

    # Make predictions using the loaded model
    predictions = model.predict(scaled_data)

    # Prepare the result as a dictionary with the company names and predictions
    results = {company: round(pred, 2) for company, pred in zip(companies, predictions[0])}

    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
