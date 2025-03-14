#!/bin/bash

# Variables
VENV_DIR=".venv"
PYTHON="$VENV_DIR/Scripts/python"

# Activate virtual environment
source "$VENV_DIR/Scripts/activate"

# Run the main script
"$PYTHON" main.py
