#!/bin/bash
# Quick setup script for Robot Arm RL project with Copilot

echo "🤖 Setting up Robot Arm RL Project with GitHub Copilot..."

# Install VS Code extensions via command line
echo "📦 Installing VS Code extensions..."
code --install-extension GitHub.copilot
code --install-extension GitHub.copilot-chat
code --install-extension ms-python.python

# Clone the project
echo "📥 Cloning project..."
read -p "Enter your GitHub username: " username
git clone https://github.com/$username/robot-arm-rl-project.git
cd robot-arm-rl-project

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install -r requirements.txt

# Open in VS Code
echo "🚀 Opening in VS Code..."
code .

echo "✅ Setup complete!"
echo "💡 To use Copilot:"
echo "   1. Press Ctrl+Shift+P and type 'GitHub Copilot: Sign In'"
echo "   2. Open enhanced_trainer.py and start coding"
echo "   3. Press Ctrl+Shift+I for Copilot Chat"
echo "   4. Try typing: # Create a new reward function"
