import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib

def train_model():
    # Load dataset
    dataset_path = "nutrigenomics_dataset.csv"
    df = pd.read_csv(dataset_path)

    # Target column
    target = "Nutrient_Name"

    # Print raw dataset before encoding
    print("\n📊 Raw Training Data Before Encoding:\n", df.head())

    # Identify categorical columns (excluding the target column)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object' and col != target]
    
    # ✅ FIX: Replace NaN values and ensure "None" is a valid category
    for col in categorical_columns:
        df[col] = df[col].fillna("None").astype(str)  # Ensure 'None' is a valid category

    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)  

    # Print features after encoding
    print("\n🔍 Features After One-Hot Encoding in Training:", df_encoded.columns.tolist())

    # Encode target column
    label_encoder = LabelEncoder()
    df_encoded[target] = label_encoder.fit_transform(df_encoded[target])

    # Print unique target class mappings
    print("\n✅ Target Class Mappings:", dict(enumerate(label_encoder.classes_)))

    # Split the data
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    # Print final features used in training
    print("\n✅ Final Features Used in Training:", X.columns.tolist())

    # Train-test split (Stratified to balance class distribution)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the XGBoost model with better hyperparameters
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(pd.get_dummies(y_test), model.predict_proba(X_test), multi_class='ovr')

    print(f"\n✅ Model Accuracy: {accuracy:.2f}")
    print(f"✅ ROC-AUC Score: {auc:.2f}")

    # Save the model
    joblib.dump(model, "nutrigenomics_model.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    joblib.dump(list(X.columns), "feature_columns.pkl")

if __name__ == "__main__":
    train_model()
