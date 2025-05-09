# Save this as convert_model.py
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# Load your existing model
print("Loading existing model...")
model = joblib.load("catboost_model.pkl")

# Save it again with protocol=4 (more compatible)
print("Re-saving model with more compatible format...")
joblib.dump(model, "catboost_model_new.pkl", compress=True, protocol=4)

# Rename the file
import os
os.rename("catboost_model_new.pkl", "catboost_model.pkl")
print("Model converted successfully!")