from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

# Load the trained model
model = joblib.load("churn_model.joblib")

# Initialize Flask app
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Create a feature list in the order used during training
        features = [
            data['account_length'],
            data['number_vmail_messages'],
            data['total_day_minutes'],
            data['total_day_calls'],
            data['total_eve_minutes'],
            data['total_eve_calls'],
            data['total_night_minutes'],
            data['total_night_calls'],
            data['total_intl_minutes'],
            data['total_intl_calls'],
            data['customer_service_calls'],
            data['international_plan'],
            data['voice_mail_plan'],
            data['state_0'],
            data['state_1'],
            data['state_2'],
            data['state_3'],
            data['state_4'],
            data['state_5'],
            data['state_6'],
        ]
        X = np.array(features).reshape(1, -1)
        prediction = model.predict(X)[0]
        result = "Customer Will Churn" if prediction == 1 else "Customer Will Not Churn"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Create a folder for templates if it doesn't exist
if not os.path.exists('templates'):
    os.makedirs('templates')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)