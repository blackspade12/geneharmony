import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

from sklearn.preprocessing import LabelEncoder

def train_model():
    # Load dataset
    dataset_path = "nutrigenomics_dataset.csv"
    df = pd.read_csv(dataset_path)

    # Target column and features
    target = "Nutrient_Name"

    # Encode categorical variables (excluding the target column)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object' and col != target]
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Debug: Confirm target column is intact
    print("Columns after encoding:", df_encoded.columns)

    # Encode target column
    label_encoder = LabelEncoder()
    df_encoded[target] = label_encoder.fit_transform(df[target])

    # Debug: Check unique classes and mappings
    print("Classes:", label_encoder.classes_)

    # Split the data
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBoost model
    # Train the XGBoost model
    model = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss')
    model.fit(X_train, y_train)


    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(pd.get_dummies(y_test), model.predict_proba(X_test), multi_class='ovr')

    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"ROC-AUC: {auc:.2f}")

    # Save the model
    joblib.dump(model, "nutrigenomics_model.pkl")
    # Save the label encoder
    joblib.dump(label_encoder, "label_encoder.pkl")
    joblib.dump(list(X.columns), "feature_columns.pkl")




if __name__ == "__main__":
    train_model()