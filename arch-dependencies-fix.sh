#!/bin/bash
# Quick Fix Script dla Arch Linux + Agent Zero V1
# NLU Task Decomposer Installation

echo "🚀 Agent Zero V1 - NLU Task Decomposer Quick Fix"
echo "=================================================="

# 1. Install system packages on Arch
echo "📦 Installing system packages..."
sudo pacman -S --needed python-click python-rich python-aiohttp python-pip

# 2. Install spaCy via pipx or system pip with break-system-packages
echo "🔤 Installing spaCy and model..."
pip install --user --break-system-packages spacy
python -m spacy download en_core_web_sm --user

# 3. Alternative: Use virtual environment
echo "🐍 Creating virtual environment (recommended)..."
python -m venv venv_nlu
source venv_nlu/bin/activate
pip install spacy click rich aiohttp asyncio-extras
python -m spacy download en_core_web_sm

echo "✅ Dependencies installed!"
echo ""
echo "📋 Next steps:"
echo "1. Activate venv: source venv_nlu/bin/activate"
echo "2. Run the corrected Python scripts"
echo "3. Test the system"