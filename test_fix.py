import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def test_fix():
    print("Testing the categorical variable fix...")
    
    # Load data
    df = pd.read_csv("data/applications_sample.csv")
    print(f"Loaded data with shape: {df.shape}")
    
    # Prepare features
    y = df["approved"].astype(int).values
    feature_df = df.drop(columns=["approved", "applicant_id", "name"])
    
    # Handle categorical variables
    categorical_columns = feature_df.select_dtypes(include=['object']).columns
    print(f"Found categorical columns: {categorical_columns.tolist()}")
    
    if len(categorical_columns) > 0:
        feature_df = pd.get_dummies(feature_df, columns=categorical_columns, drop_first=True)
        print(f"After encoding, shape: {feature_df.shape}")
        print(f"New columns: {feature_df.columns.tolist()}")
    
    X = feature_df.values
    print(f"Final X shape: {X.shape}, dtype: {X.dtype}")
    
    # Train a simple model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"✅ SUCCESS! Model trained successfully with AUC: {auc:.3f}")
    print("The categorical variable issue has been fixed!")
    
    return True

if __name__ == "__main__":
    try:
        test_fix()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
