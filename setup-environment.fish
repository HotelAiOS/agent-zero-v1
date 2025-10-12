#!/usr/bin/env fish

# Agent Zero V2.0 Environment Setup for Arch Linux + Fish Shell
# This script properly handles the environment setup and dependency installation

echo "🔧 Agent Zero V2.0 - Environment Setup (Arch Linux + Fish Shell)"
echo "=================================================================="

# Check Fish shell
if not type -q fish
    echo "❌ Fish shell not detected. This script requires Fish shell."
    exit 1
end

echo "✅ Fish shell detected"

# Check Python version
set python_version (python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "🐍 Python version: $python_version"

# Method selection
echo ""
echo "Select installation method:"
echo "1. Virtual Environment (Recommended)"
echo "2. System packages (Arch pacman)"
echo "3. Pipenv (Modern Python workflow)"

read -P "Choose method [1-3]: " method

switch $method
    case "1"
        echo "📦 Setting up Virtual Environment..."
        
        # Remove existing venv if corrupted
        if test -d venv
            echo "🗑️  Removing existing venv..."
            rm -rf venv
        end
        
        # Create fresh venv
        echo "🔨 Creating new virtual environment..."
        python -m venv venv
        
        # Fish-specific activation check
        if test -f venv/bin/activate.fish
            echo "✅ Fish activation script found"
            
            # Activate venv in Fish way
            echo "🔌 Activating virtual environment..."
            source venv/bin/activate.fish
            
            # Upgrade pip
            echo "📦 Upgrading pip..."
            pip install --upgrade pip
            
            # Install core dependency first
            echo "📊 Installing NetworkX..."
            pip install "networkx>=3.0"
            
            # Install other requirements
            echo "📚 Installing additional requirements..."
            pip install asyncio sqlite3 requests pydantic
            
            echo "✅ Virtual environment setup complete!"
            echo "💡 To activate in future Fish sessions:"
            echo "   source venv/bin/activate.fish"
            
        else
            echo "❌ Fish activation script not found. Using alternative method..."
            
            # Alternative: direct pip usage
            echo "📦 Installing dependencies directly to venv..."
            venv/bin/pip install --upgrade pip
            venv/bin/pip install "networkx>=3.0" asyncio sqlite3 requests pydantic
            
            echo "✅ Dependencies installed to venv!"
            echo "💡 Use: venv/bin/python for execution"
        end
    
    case "2"
        echo "🏛️  Installing system packages via pacman..."
        
        # Core Python packages
        sudo pacman -S --needed python-networkx python-requests python-pydantic python-pytest
        
        # Optional but useful
        sudo pacman -S --needed python-numpy python-pandas python-aiohttp
        
        echo "✅ System packages installed!"
        echo "💡 Use system Python: python"
    
    case "3"
        echo "🔧 Setting up Pipenv environment..."
        
        # Install pipenv if not present
        if not type -q pipenv
            echo "📦 Installing pipenv..."
            sudo pacman -S python-pipenv
        end
        
        # Initialize Pipfile if not exists
        if not test -f Pipfile
            echo "📝 Creating Pipfile..."
            pipenv install
        end
        
        # Install dependencies
        echo "📚 Installing dependencies..."
        pipenv install "networkx>=3.0" requests pydantic pytest asyncio
        
        echo "✅ Pipenv environment ready!"
        echo "💡 To activate: pipenv shell"
        echo "💡 To run: pipenv run python script.py"
    
    case "*"
        echo "❌ Invalid option. Exiting."
        exit 1
end

echo ""
echo "🧪 Testing installation..."

# Test import based on method
switch $method
    case "1"
        if test -f venv/bin/activate.fish
            # Test with activated venv
            python -c "
import networkx as nx
import sqlite3
import asyncio
print('✅ All core dependencies imported successfully!')
print(f'NetworkX version: {nx.__version__}')
"
        else
            # Test with venv python directly
            venv/bin/python -c "
import networkx as nx
import sqlite3
import asyncio
print('✅ All core dependencies imported successfully!')
print(f'NetworkX version: {nx.__version__}')
"
        end
    
    case "2"
        python -c "
import networkx as nx
import sqlite3
import asyncio
print('✅ All core dependencies imported successfully!')
print(f'NetworkX version: {nx.__version__}')
"
    
    case "3"
        pipenv run python -c "
import networkx as nx
import sqlite3
import asyncio
print('✅ All core dependencies imported successfully!')
print(f'NetworkX version: {nx.__version__}')
"
end

echo ""
echo "🎯 Now test the DynamicWorkflowOptimizer:"

switch $method
    case "1"
        echo "   venv/bin/python shared/orchestration/dynamic_workflow_optimizer.py"
    case "2" 
        echo "   python shared/orchestration/dynamic_workflow_optimizer.py"
    case "3"
        echo "   pipenv run python shared/orchestration/dynamic_workflow_optimizer.py"
end

echo ""
echo "🎉 Agent Zero V2.0 environment setup complete!"