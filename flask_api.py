from flask import Flask, request, send_file
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

#Loading my clustering model and label mapping
model = joblib.load('customer_segmentation_model.pkl')
cluster_labels = {
    0: "High-Spending Loyal Customers",
    1: "New Low-Spending Buyers",
    2: "Frequent Moderate-Spenders"
}

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    data = pd.read_csv(file)
    
# Predict for uploaded CSV
@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    file = request.files['file']
    if file:
        df = pd.read_csv(file)

        # Ensure columns are in the same order as model was trained on
        columns = ['recency', 'purchase_frequency', 'avg_order_value', 'total_spend', 'age', 'purchase_duration']
        df = df[columns]

        predictions = model.predict(df)
        df['Predicted Segment'] = predictions

        # Display first few predictions
        output = df[['Predicted Segment']].head().to_html(index=False)

        return render_template('index.html', prediction_text="CSV Predictions:", prediction_result=output)
    return render_template('index.html', prediction_text="No file uploaded.")

# Predict for single customer input
@app.route('/predict_single', methods=['POST'])
def predict_single():
    try:
        # Get form inputs
        recency = float(request.form['recency'])
        purchased_frequency = float(request.form['purchase_frequency'])
        avg_order_value = float(request.form['avg_order_value'])
        total_spend = float(request.form['total_spend'])
        age = float(request.form['age'])
        purchase_duration = float(request.form['purchase_duration'])

        # Build input DataFrame
        input_df = pd.DataFrame([[recency, purchase_frequency, avg_order_value, total_spend, age, purchase_duration]],
                                columns=['Recency', 'PurchaseFrequency', 'AverageOrderValue', 'TotalSpend', 'Age', 'PurchaseDuration'])

        prediction = model.predict(input_df)[0]

        return render_template('index.html', single_prediction=f"The predicted segment is: {prediction}")
    except Exception as e:
        return render_template('index.html', single_prediction=f"Error: {str(e)}")

# Run the app
if _name_ == '_main_':
    app.run(debug=True)
