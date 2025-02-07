from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import json  # For debugging JSON input

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

def load_model():
    """Loads the trained model, label encoder, and feature columns safely."""
    try:
        model = joblib.load("nutrigenomics_model.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        return model, label_encoder, feature_columns
    except FileNotFoundError as e:
        raise FileNotFoundError("❌ Missing required model files: " + str(e))

def process_input(data, feature_columns):
    """
    Processes user input into a DataFrame with the same features as the training set.
    Ensures all required columns are present and unseen categories don't break the model.
    """
    
    # Debug: Print Raw JSON Data to ensure API receives full input
    print("\n📥 Raw API Input Data:", json.dumps(data, indent=2))
    
    # Convert JSON to DataFrame
    df_input = pd.DataFrame([data])
    
    # Debug: Print DataFrame before encoding
    print("\n📊 Converted DataFrame Before Encoding:\n", df_input)

    # Define categorical columns (should match training.py)
    categorical_columns = ["SNP_ID", "Gene_Name", "SNP_Nutrient_Interaction", "Gender", 
                           "Ethnicity", "Health_Conditions", "Dietary_Preferences", "Alleles", "Genotype"]

    # Ensure categorical columns are present and convert them to strings
    for col in categorical_columns:
        if col in df_input.columns:
            df_input[col] = df_input[col].fillna("None").astype(str)

    # Debug: Print features before encoding
    print("\n🔍 Features Before One-Hot Encoding:", df_input.columns.tolist())

    # Apply one-hot encoding
    df_input = pd.get_dummies(df_input)

    # Debug: Print features after encoding
    print("\n✅ Features After One-Hot Encoding:", df_input.columns.tolist())

    # Ensure all required feature columns exist
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0  # Set missing categories to 0

    # Ensure Sample_ID exists if required by model
    if "Sample_ID" in feature_columns and "Sample_ID" not in df_input.columns:
        df_input["Sample_ID"] = 0  # Assign a default value

    # Set a default Effect_Size if missing
    if "Effect_Size" in df_input.columns and df_input["Effect_Size"].sum() == 0:
        df_input["Effect_Size"] = 0.5  # Assign a reasonable default

    # Ensure columns match training order
    df_input = df_input[feature_columns]

    # Debug: Print final processed input
    print("\n📊 Final Processed Input for Model Prediction:\n", df_input.head())

    return df_input

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint to predict the best nutrient for a user based on input data."""
    try:
        model, label_encoder, feature_columns = load_model()
        input_data = request.json
        
        # Debug: Ensure input is received
        if not input_data:
            return jsonify({"status": "error", "message": "No input data received"}), 400
        
        X_input = process_input(input_data, feature_columns)

        # Debug: Check for feature mismatch
        if list(X_input.columns) != feature_columns:
            return jsonify({
                "status": "error",
                "message": f"Feature mismatch: Expected {feature_columns}, but got {list(X_input.columns)}"
            }), 400

        # Model probability predictions
        probabilities = model.predict_proba(X_input)
        prediction_index = model.predict(X_input)
        predicted_nutrient = label_encoder.inverse_transform(prediction_index)

        # Debug: Log prediction probabilities
        print("\n🔍 Prediction Probabilities:", probabilities)
        print("🔍 Predicted Index:", prediction_index)
        print("✅ Predicted Nutrient:", predicted_nutrient)

        return jsonify({
            "status": "success",
            "prediction": predicted_nutrient[0],
            "probabilities": probabilities.tolist()  # Debugging output
        })
    except Exception as e:
        # Enhanced error reporting
        error_message = f"Error: {str(e)}"
        print("🚨", error_message)
        return jsonify({"status": "error", "message": error_message}), 500


@app.route("/recommend", methods=["POST"])
def recommend():
    """Endpoint to predict and provide dietary recommendations."""
    try:
        model, label_encoder, feature_columns = load_model()
        input_data = request.json
        
        # Debug: Ensure input is received
        if not input_data:
            return jsonify({"status": "error", "message": "No input data received"}), 400
        
        X_input = process_input(input_data, feature_columns)

        # Model prediction
        prediction = model.predict(X_input)
        predicted_nutrient = label_encoder.inverse_transform(prediction)[0]

        # Nutrient recommendations
        recommendations = {
            "Calcium": "Include dairy products, leafy greens, and fortified cereals.",
            "Iron": "Consume lean meats, beans, and spinach for iron-rich nutrition.",
            "Omega-3": "Add fatty fish, walnuts, and flaxseeds to your diet.",
            "Vitamin D": "Spend time in sunlight and eat fortified foods or fatty fish.",
            "Zinc": "Include nuts, seeds, and whole grains for optimal zinc intake.",
            "Magnesium": "Eat nuts, whole grains, and leafy greens for better magnesium levels.",
            "Folate": "Consume legumes, citrus fruits, and leafy vegetables.",
            "Vitamin B12": "Eat eggs, dairy, and fortified foods for B12 needs."
        }

        # Provide recommendation or a fallback message
        recommendation = recommendations.get(
            predicted_nutrient, 
            f"No specific recommendation available for {predicted_nutrient}. Consider consulting a nutritionist."
        )

        return jsonify({
            "status": "success",
            "prediction": predicted_nutrient,
            "recommendation": recommendation
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
