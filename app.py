from flask import Flask, request, jsonify, render_template
import os
import sys
import traceback
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize model and scaler as None
model = None
scaler = None

# Try to load the models with error handling
try:
    # First, ensure we're importing these after NumPy is properly installed
    import numpy as np
    import pandas as pd
    import joblib
    
    print("Loading models...")
    # Try to load the model file
    model = joblib.load("catboost_model.pkl")
    scaler = joblib.load("standard_scaler.pkl")
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    traceback.print_exc()

# API endpoint for health check
@app.route("/api", methods=["GET"]) 
def api_home():
    model_status = "loaded" if model is not None else "not loaded"
    scaler_status = "loaded" if scaler is not None else "not loaded"
    return f"Crop Recommendation API is running! Model: {model_status}, Scaler: {scaler_status}"

# API endpoint for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if model is available
        if model is None:
            return jsonify({
                "error": "CatBoost model not loaded. The server is running in limited mode.",
                "status": "failure"
            }), 500
        
        # Import these here to ensure they're only used if model is loaded
        import numpy as np
        import pandas as pd
        
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
        
        # Scale the data if scaler is available
        if scaler is not None:
            input_scaled = scaler.transform(input_df[required_features])
        else:
            input_scaled = input_df[required_features].values
        
        # Make prediction
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
        traceback.print_exc()
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
            
            # Scale the data before prediction
            if scaler is not None:
                input_scaled = scaler.transform(input_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']])
            else:
                input_scaled = input_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].values
            
            # Make prediction with scaled data
            prediction = model.predict(input_scaled)[0]
            
            # Convert numpy type to Python native type
            if isinstance(prediction, np.ndarray):
                prediction = prediction.tolist()
            elif hasattr(prediction, 'item'):
                prediction = prediction.item()
            else:
                prediction = str(prediction)
                
            try:
                # Try to render the template
                return render_template('results.html', prediction=prediction, input_data=input_data)
            except:
                # Fall back to JSON response if template not found
                return jsonify({
                    "prediction": prediction, 
                    "input_data": input_data,
                    "status": "success"
                })
        
        except Exception as e:
            error_message = str(e)
            try:
                return render_template('index.html', error=error_message)
            except:
                return jsonify({"error": error_message, "status": "failure"})
    else:
        # Try to render the form template
        try:
            return render_template('index.html')
        except:
            # Return API information if templates are not available
            return """
            <h1>Crop Recommendation API</h1>
            <p>This is a REST API for crop recommendations. Make POST requests to /predict with soil parameters.</p>
            <p>Example POST body to /predict:</p>
            <pre>
            {
                "N": 90,
                "P": 42,
                "K": 43,
                "temperature": 20.879744,
                "humidity": 82.002744,
                "ph": 6.502985,
                "rainfall": 202.935536
            }
            </pre>
            """

if __name__ == "__main__":
    app.run(debug=True)