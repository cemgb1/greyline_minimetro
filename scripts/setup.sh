#!/bin/bash
# Setup script for Mini Metro RL on GCP
# Installs dependencies and configures the environment

set -e

echo "🚇 Setting up Mini Metro RL Environment"
echo "======================================"

# Check if running on GCP
if [ -f /etc/google_compute_engine ]; then
    echo "✓ Detected Google Cloud Platform instance"
    IS_GCP=true
else
    echo "⚠ Not running on GCP, proceeding with local setup"
    IS_GCP=false
fi

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3-pip \
    git \
    htop \
    tmux \
    vim \
    curl \
    wget \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqtgui4 \
    libqtwebkit4 \
    libqt4-test \
    python3-pyqt5 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

# Install SDL2 for pygame
echo "🎮 Installing SDL2 for pygame..."
sudo apt-get install -y \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libfreetype6-dev \
    libportmidi-dev

# Create project directory
PROJECT_DIR="$HOME/mini_metro_rl"
echo "📁 Setting up project directory: $PROJECT_DIR"

if [ ! -d "$PROJECT_DIR" ]; then
    mkdir -p "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"

# Clone repository if not already present
if [ ! -d ".git" ]; then
    echo "📥 Cloning repository..."
    git clone https://github.com/cemgb1/greyline_minimetro.git .
fi

# Create Python virtual environment
echo "🐍 Creating Python virtual environment..."
python3.9 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Install PyTorch with CUDA support if available
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected, installing PyTorch with CUDA support"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "⚠ No NVIDIA GPU detected, installing CPU-only PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Create necessary directories
echo "📂 Creating project directories..."
mkdir -p logs/tensorboard
mkdir -p models
mkdir -p data
mkdir -p evaluation_results
mkdir -p checkpoints

# Set up environment variables
echo "🌍 Setting up environment variables..."
cat >> ~/.bashrc << 'EOF'

# Mini Metro RL Environment
export MINI_METRO_RL_HOME="$HOME/mini_metro_rl"
export PYTHONPATH="$MINI_METRO_RL_HOME:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
alias metro-rl="cd $MINI_METRO_RL_HOME && source venv/bin/activate"
alias metro-train="cd $MINI_METRO_RL_HOME && source venv/bin/activate && python train.py"
alias metro-eval="cd $MINI_METRO_RL_HOME && source venv/bin/activate && python evaluate.py"
alias metro-viz="cd $MINI_METRO_RL_HOME && source venv/bin/activate && python main.py --visualize"
EOF

# Install additional monitoring tools for GCP
if [ "$IS_GCP" = true ]; then
    echo "☁️ Installing GCP monitoring tools..."
    
    # Install Cloud SDK if not present
    if ! command -v gcloud &> /dev/null; then
        echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
        curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
        sudo apt-get update && sudo apt-get install google-cloud-cli
    fi
    
    # Install Cloud Logging agent
    curl -sSO https://dl.google.com/cloudagents/add-logging-agent-repo.sh
    sudo bash add-logging-agent-repo.sh --also-install
    
    # Configure logging
    sudo tee /etc/google-fluentd/config.d/mini-metro-rl.conf > /dev/null <<EOF
<source>
  @type tail
  format none
  path $PROJECT_DIR/logs/*.log
  pos_file /var/lib/google-fluentd/pos/mini-metro-rl.log.pos
  read_from_head true
  tag mini-metro-rl
</source>
EOF
    
    sudo service google-fluentd restart
fi

# Create systemd service for training
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/mini-metro-rl.service > /dev/null <<EOF
[Unit]
Description=Mini Metro RL Training Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
ExecStart=$PROJECT_DIR/venv/bin/python train.py --config dqn_config
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mini-metro-rl.service

# Test installation
echo "🧪 Testing installation..."
source venv/bin/activate

# Test imports
python -c "
import torch
import gymnasium
import pygame
import numpy as np
import matplotlib
import pandas as pd
import yaml
print('✓ All major dependencies imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"

# Test basic functionality
echo "🎯 Testing basic functionality..."
python -c "
import sys
sys.path.insert(0, '.')
from mini_metro_rl.src.utils.config import load_config
from mini_metro_rl.src.environment.metro_env import MetroEnvironment, EnvironmentConfig

config = load_config('dqn_config')
env_config = EnvironmentConfig()
env = MetroEnvironment(env_config)
obs, info = env.reset()
print(f'✓ Environment created successfully')
print(f'✓ Observation space: {env.observation_space.shape}')
print(f'✓ Action space: {env.action_space}')
"

# Performance recommendations
echo ""
echo "🚀 Setup Complete!"
echo "=================="
echo ""
echo "📋 Next Steps:"
echo "1. Source your environment: source ~/.bashrc"
echo "2. Activate virtual environment: metro-rl"
echo "3. Run training: metro-train --config dqn_config"
echo "4. Monitor with: sudo systemctl status mini-metro-rl"
echo ""
echo "💡 Performance Tips:"
echo "- Use 'tmux' for persistent sessions"
echo "- Monitor GPU usage with 'nvidia-smi'"
echo "- Check logs in logs/ directory"
echo "- Use TensorBoard for training visualization"
echo ""

if [ "$IS_GCP" = true ]; then
    echo "☁️ GCP-specific commands:"
    echo "- View logs: gcloud logging read 'resource.type=gce_instance'"
    echo "- SSH tunnel for TensorBoard: gcloud compute ssh [INSTANCE] -- -L 6006:localhost:6006"
    echo ""
fi

echo "🎮 Ready to train some Metro RL agents!"
echo "Documentation: See README.md for detailed usage instructions"