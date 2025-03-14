#!/bin/bash

# Variables
VENV_DIR="../.venv"
PYTHON="$VENV_DIR/Scripts/python"
PIP="$VENV_DIR/Scripts/pip"

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting setup...${NC}"

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${GREEN}Creating virtual environment...${NC}"
    python -m venv "$VENV_DIR"
    echo -e "${GREEN}Upgrading pip...${NC}"
    source "$VENV_DIR/Scripts/activate"
    "$PIP" install --upgrade pip
    echo -e "${GREEN}Installing dependencies from requirements.txt...${NC}"
    "$PIP" install -r ../requirements.txt
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
    source "$VENV_DIR/Scripts/activate"
fi

# Prompt for API key and create .env file
if [ ! -f "../.env" ]; then
    read -p "Enter your OPENAI_API_KEY: " api_key
    echo -e "${GREEN}Creating .env file with provided API key...${NC}"
    echo "OPENAI_API_KEY=\"$api_key\"" > ../.env
else
    echo -e "${GREEN}.env file already exists.${NC}"
fi

# Install dependencies and set up pre-commit hooks
echo -e "${GREEN}Installing pre-commit hooks...${NC}"
"$PIP" install pre-commit
pre-commit install

# Verify pip and Python versions
echo -e "${GREEN}Verifying pip and Python versions...${NC}"
"$PIP" --version
"$PYTHON" --version

echo -e "${GREEN}Setup complete. You can now run the following commands:${NC}"
echo -e "${BLUE}  source $VENV_DIR/Scripts/activate  # Activate the virtual environment${NC}"
#echo "  ./scripts/run.sh                   # Run the main script"
#echo "  ./scripts/test.sh                  # Run tests"
