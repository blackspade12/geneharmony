from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

# Initialize the app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the trained model and label encoder
model = joblib.load("nutrigenomics_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define a helper function for processing user input
def process_input(data, feature_columns):
    """
    Processes user input data into a DataFrame with the same features as the training set.
    Missing columns are filled with 0.
    """
    df_input = pd.DataFrame([data])
    df_input = pd.get_dummies(df_input)
    missing_cols = set(feature_columns) - set(df_input.columns)
    for col in missing_cols:
        df_input[col] = 0  # Add missing columns with default values
    return df_input[feature_columns]  # Ensure correct column order

# Define the prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse user input
        input_data = request.json
        
        # Load feature columns from the training process
        feature_columns = joblib.load("feature_columns.pkl")

        # Process the input
        X_input = process_input(input_data, feature_columns)

        # Predict with the model
        prediction = model.predict(X_input)
        predicted_nutrient = label_encoder.inverse_transform(prediction)

        return jsonify({
            "status": "success",
            "prediction": predicted_nutrient[0]
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Define a recommendation endpoint
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        # Parse user input
        input_data = request.json

        # Load feature columns from the training process
        feature_columns = joblib.load("feature_columns.pkl")

        # Process the input
        X_input = process_input(input_data, feature_columns)

        # Predict with the model
        prediction = model.predict(X_input)
        predicted_nutrient = label_encoder.inverse_transform(prediction)[0]

        # Generate recommendations based on prediction
        recommendations = {
            "Calcium": "Include dairy products, leafy greens, and fortified cereals.",
            "Iron": "Consume lean meats, beans, and spinach for iron-rich nutrition.",
            "Omega-3": "Add fatty fish, walnuts, and flaxseeds to your diet.",
            "Vitamin D": "Spend time in sunlight and eat fortified foods or fatty fish.",
            "Zinc": "Include nuts, seeds, and whole grains for optimal zinc intake."
        }
        recommendation = recommendations.get(predicted_nutrient, "No specific recommendation available.")

        return jsonify({
            "status": "success",
            "prediction": predicted_nutrient,
            "recommendation": recommendation
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
