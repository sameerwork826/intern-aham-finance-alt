import pandas as pd
import numpy as np

def debug_data_loading():
    print("Loading data...")
    df = pd.read_csv("data/applications_sample.csv")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types: {df.dtypes}")
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nChecking for categorical columns...")
    feature_df = df.drop(columns=["approved", "applicant_id", "name"])
    categorical_columns = feature_df.select_dtypes(include=['object']).columns
    print(f"Categorical columns: {categorical_columns.tolist()}")
    
    if len(categorical_columns) > 0:
        print(f"\nUnique values in categorical columns:")
        for col in categorical_columns:
            print(f"{col}: {feature_df[col].unique()}")
        
        print("\nApplying one-hot encoding...")
        feature_df_encoded = pd.get_dummies(feature_df, columns=categorical_columns, drop_first=True)
        print(f"After encoding shape: {feature_df_encoded.shape}")
        print(f"New columns: {feature_df_encoded.columns.tolist()}")
        
        print("\nConverting to numpy array...")
        X = feature_df_encoded.values
        print(f"X shape: {X.shape}")
        print(f"X dtype: {X.dtype}")
        print(f"Sample values: {X[:2]}")
        
        return X
    else:
        print("No categorical columns found")
        X = feature_df.values
        print(f"X shape: {X.shape}")
        return X

if __name__ == "__main__":
    try:
        X = debug_data_loading()
        print("\nSUCCESS: Data preprocessing completed without errors!")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
