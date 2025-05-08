from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
model = joblib.load("catboost_model.pkl")
scaler = joblib.load("standard_scaler.pkl")  # Add this line to load the scaler

# API endpoint for health check
@app.route("/api", methods=["GET"]) 
def api_home():
    return "Crop Recommendation API is running!"

# API endpoint for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate input features
        required_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        # Create DataFrame with features
        input_df = pd.DataFrame([data])
        
        # Scale the data before prediction (add this step)
        input_scaled = scaler.transform(input_df[required_features])
        
        # Make prediction with scaled data
        prediction = model.predict(input_scaled)[0]
        
        # Convert numpy type to Python native type
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
        elif hasattr(prediction, 'item'):
            prediction = prediction.item()
        else:
            prediction = str(prediction)
            
        return jsonify({
            "prediction": prediction,
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failure"
        }), 500

# Add this new endpoint to your app.py
@app.route("/predict_top", methods=["POST"])
def predict_top():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate input features
        required_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        # Create DataFrame with features
        input_df = pd.DataFrame([data])
        
        # Scale the data
        input_scaled = scaler.transform(input_df[required_features])
        
        # Get prediction probabilities if supported
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(input_scaled)[0]
            # Get top 3 predictions
            indices = np.argsort(probs)[::-1][:3]
            top_crops = []
            
            for i in indices:
                crop = model.classes_[i]
                confidence = float(probs[i] * 100)  # Convert to native float and percentage
                top_crops.append({"crop": crop, "confidence": round(confidence, 2)})
            
            return jsonify({
                "predictions": top_crops,
                "status": "success"
            }), 200
        else:
            # If model doesn't support probabilities, return just the prediction
            prediction = model.predict(input_scaled)[0]
            if isinstance(prediction, np.ndarray):
                prediction = prediction.tolist()
            elif hasattr(prediction, 'item'):
                prediction = prediction.item()
            else:
                prediction = str(prediction)
                
            return jsonify({
                "predictions": [{"crop": prediction, "confidence": 100}],
                "status": "success"
            }), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failure"
        }), 500

# Web interface routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get user inputs from the form
            N = float(request.form.get('N'))
            P = float(request.form.get('P'))
            K = float(request.form.get('K'))
            temperature = float(request.form.get('temperature'))
            humidity = float(request.form.get('humidity'))
            ph = float(request.form.get('ph'))
            rainfall = float(request.form.get('rainfall'))

            # Create input data in the same format as the API
            input_data = {
                'N': N,
                'P': P,
                'K': K,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall
            }
            
            # Create DataFrame with features
            input_df = pd.DataFrame([input_data])
            
            # Scale the data before prediction (add this step)
            input_scaled = scaler.transform(input_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']])
            
            # Make prediction with scaled data
            prediction = model.predict(input_scaled)[0]
            
            # Convert numpy type to Python native type
            if isinstance(prediction, np.ndarray):
                prediction = prediction.tolist()
            elif hasattr(prediction, 'item'):
                prediction = prediction.item()
            else:
                prediction = str(prediction)
                
            # Render the results template
            return render_template('results.html', prediction=prediction, input_data=input_data)
        
        except Exception as e:
            error_message = str(e)
            return render_template('index.html', error=error_message)
    else:
        # Render the input form template
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)