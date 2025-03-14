#!/bin/bash

# Variables
VENV_DIR="../.venv"

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting cleanup...${NC}"

# Prompt for confirmation
read -p "Are you sure you want to remove the virtual environment and clean up cache files? (y/n): " confirm
if [[ "$confirm" != [yY] ]]; then
    echo -e "${RED}Cleanup aborted.${NC}"
    exit 0
fi

# Clean up virtual environment
if [ -d "$VENV_DIR" ]; then
    echo -e "${GREEN}Removing virtual environment...${NC}"
    rm -rf "$VENV_DIR"
    echo -e "${GREEN}Virtual environment removed.${NC}"
else
    echo -e "${RED}Virtual environment does not exist.${NC}"
fi

# Clean up Python cache files
echo -e "${GREEN}Removing Python cache files...${NC}"
find .. -type f -name '*.pyc' -delete
find .. -type d -name '__pycache__' -delete
echo -e "${GREEN}Python cache files removed.${NC}"

echo -e "${GREEN}Cleanup complete.${NC}"
