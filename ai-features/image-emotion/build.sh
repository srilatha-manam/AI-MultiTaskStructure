#!/bin/bash

echo "Building Image Emotion Detection Service..."

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs
mkdir -p artifacts

echo "Image Emotion Detection Service build completed!"

# Run the service
echo "Starting the service..."
python main.py