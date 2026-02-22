import joblib
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load trained model and scaler
try:
    model = joblib.load('logistic_regression_model.joblib')
    scaler = joblib.load('scaler.joblib')
except Exception as e:
    raise Exception(f"Error loading model or scaler: {e}")

# Initialize Flask app


# Expected feature order (must match training order)
EXPECTED_FEATURES = ['Vrms', 'Irms', 'Energy', 'Frequency', 'PF']


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        # Check if all required features are present
        missing = [feature for feature in EXPECTED_FEATURES if feature not in data]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 400

        # Convert to DataFrame (maintain correct order)
        features = pd.DataFrame([[data[feature] for feature in EXPECTED_FEATURES]],
                                columns=EXPECTED_FEATURES)

        # Scale input
        scaled_features = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)

        # Return result
        return jsonify({
            'input': data,
            'prediction': int(prediction[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 
