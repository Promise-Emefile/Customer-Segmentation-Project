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

    features = data[['recency', 'purchase_frequency', 'avg_order_value', 'total_spend', 'purchase_duration']]
    predictions = model.predict(features)
    data['cluster'] = predictions
    data['cluster_label'] = data['cluster'].map(cluster_labels)

    return data.to_html(classes='table table-striped')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)



