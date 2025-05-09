#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies in exact order with specific versions
pip install --upgrade pip
pip install numpy==1.23.5  # Use older, more compatible version
pip install scipy==1.10.1  # Add explicit scipy version
pip install joblib==1.2.0
pip install scikit-learn==1.3.0  # Use slightly older version
pip install catboost==1.1.1  # Use slightly older version
pip install -r requirements.txt


# Create templates directory if it doesn't exist
mkdir -p templates