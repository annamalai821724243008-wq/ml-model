from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load model and feature list
try:
    model = joblib.load("logistic_regression_model.pkl")
    feature_columns = joblib.load("model_features.pkl")
except Exception as e:
    raise Exception(f"Error loading model files: {e}")


@app.route("/")
def home():
    return jsonify({"message": "ML Prediction API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON request
        data = request.get_json()

        if data is None:
            return jsonify({"error": "No JSON data received"}), 400

        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Align columns with training features
        aligned_input = input_df.reindex(columns=feature_columns, fill_value=0)

        # Convert to numeric (important for ML models)
        aligned_input = aligned_input.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Make prediction
        prediction = model.predict(aligned_input)

        # Probability (optional but useful)
        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(aligned_input).tolist()

        return jsonify({
            "prediction": prediction.tolist(),
            "probability": probability
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
