#!/bin/bash

# Agentic Data Science Team - Installation Script

echo "🚀 Agentic Data Science Team - Installation Script"
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Python dependencies installed successfully"
else
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "⚠️  Ollama is not installed."
    echo "Please install Ollama from: https://ollama.ai/"
    echo "After installation, run: ollama pull llama3.1:latest"
    echo ""
    echo "For macOS:"
    echo "  brew install ollama"
    echo ""
    echo "For Linux:"
    echo "  curl -fsSL https://ollama.ai/install.sh | sh"
    echo ""
    echo "For Windows:"
    echo "  Download from: https://ollama.ai/download"
else
    echo "✅ Ollama found: $(ollama --version)"
    
    # Check if llama3.1 model is available
    if ollama list | grep -q "llama3.1"; then
        echo "✅ llama3.1 model is available"
    else
        echo ""
        echo "📥 Pulling llama3.1 model..."
        ollama pull llama3.1:latest
        
        if [ $? -eq 0 ]; then
            echo "✅ llama3.1 model downloaded successfully"
        else
            echo "❌ Failed to download llama3.1 model"
            exit 1
        fi
    fi
fi

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p ./data/train
mkdir -p ./data/test
mkdir -p ./logs
mkdir -p ./model
mkdir -p ./solution

echo "✅ Directories created"

# Test the setup
echo ""
echo "🧪 Testing setup..."
python3 test_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Start Ollama: ollama serve"
    echo "2. Run the team: python3 agents/agents.py"
    echo "3. Monitor progress in ./logs/ directory"
    echo "4. View results in ./solution/solution.ipynb"
    echo ""
    echo "Happy Data Science! 🚀"
else
    echo ""
    echo "⚠️  Setup test failed. Please check the errors above."
    echo "Refer to README.md for troubleshooting guidance."
    exit 1
fi 