#!/bin/bash

echo "Building Text Emotion Detection Service..."

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"

# Create necessary directories
mkdir -p logs
mkdir -p artifacts

echo "Text Emotion Detection Service build completed!"

# Run the service
echo "Starting the service..."
python main.py