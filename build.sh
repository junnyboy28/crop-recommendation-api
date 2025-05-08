#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies in the correct order
pip install --upgrade pip
pip install numpy==1.24.4  
pip install scikit-learn==1.6.0
pip install -r requirements.txt

# Create templates directory if it doesn't exist
mkdir -p templates