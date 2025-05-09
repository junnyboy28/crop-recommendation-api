# Save this as convert_model.py
import joblib
import numpy as np
import pandas as pd
import os
import traceback

try:
    print("Loading existing model...")
    model = joblib.load("catboost_model.pkl")

    # Import CatBoostClassifier after model is loaded
    # This helps if the saved model has a different version
    from catboost import CatBoostClassifier

    # Save it again with protocol=4 (more compatible)
    print("Re-saving model with more compatible format...")
    joblib.dump(model, "catboost_model_new.pkl", compress=True, protocol=4)

    # Rename the file
    os.rename("catboost_model_new.pkl", "catboost_model.pkl")
    print("Model converted successfully!")

except Exception as e:
    print(f"Error converting model: {e}")
    traceback.print_exc()