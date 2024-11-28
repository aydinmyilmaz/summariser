#!/bin/bash

# Check if the virtual environment directory exists
if [ ! -d "../venv" ]; then
  echo "Virtual environment not found. Please create it first."
  exit 1
fi

# Activate the virtual environment
source ../venv/bin/activate

# Run the Streamlit app
streamlit run main.py

# Deactivate the virtual environment after the app is closed
deactivate
