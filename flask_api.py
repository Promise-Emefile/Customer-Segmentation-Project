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
    
#defining required columns
    required_columns =[
        'recency',
        'purchase_frequency',
        'avg_order_value',
        'total_spend',
        'age',
        'purchase_duration'
    ]

    #checking for missing columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return f"""
        <h2 style ='color:red;'> ❌ Error: The following required column(s) are missing from your file:</h2>
        <ul>
            {''.join(f'<li>{col}</li>' for col in missing_columns)}
            </ul>
            """
        #selecting only the required features
    features = data[['recency', 'purchase_frequency', 'avg_order_value', 'total_spend','age', 'purchase_duration']]
    #Predict
    predictions = model.predict(features)
    data['cluster'] = predictions
    data['cluster_label'] = data['cluster'].map(cluster_labels)

    return data.to_html(classes='table table-striped')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
