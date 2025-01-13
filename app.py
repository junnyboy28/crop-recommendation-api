from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("crop_recommendation_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return "Crop Recommendation API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract input features
        features = np.array(data["features"]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        # Return the prediction
        return jsonify({
            "prediction": prediction,
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failure"
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
