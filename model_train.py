import os, json, pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from utils import ensure_dirs
from db import get_engine

MODEL_PATH = "storage/models/risk_model.pkl"
SCALER_PATH = "storage/models/scaler.pkl"

def train_model(csv_path: str = "data/applications_sample.csv"):
    ensure_dirs()
    df = pd.read_csv(csv_path)
    
    # We'll define the label as 'approved' (1/0) in the sample
    y = df["approved"].astype(int).values
    
    # Drop non-feature columns
    feature_df = df.drop(columns=["approved", "applicant_id", "name"])
    
    # Handle categorical variables by one-hot encoding
    categorical_columns = feature_df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        # One-hot encode categorical variables
        feature_df = pd.get_dummies(feature_df, columns=categorical_columns, drop_first=True)
    
    # Convert to numpy array
    X = feature_df.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Try XGBoost, fallback to Logistic Regression
    model = None
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42)
        model.fit(X_train, y_train)
    except Exception:
        model = LogisticRegression(max_iter=200, class_weight="balanced")
        model.fit(X_train, y_train)

    # Evaluate
    proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    auc = roc_auc_score(y_test, proba)
    print(f"AUC: {auc:.3f}")
    try:
        preds = (proba>=0.5).astype(int)
        print(classification_report(y_test, preds))
    except Exception:
        pass

    # Save feature names for later use
    feature_names = feature_df.columns.tolist()
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }
    
    with open(MODEL_PATH, "wb") as f: 
        pickle.dump(model_data, f)
    with open(SCALER_PATH, "wb") as f: 
        pickle.dump(scaler, f)
    
    print(f"Model trained with {len(feature_names)} features: {feature_names}")
    return auc

if __name__ == "__main__":
    train_model()
