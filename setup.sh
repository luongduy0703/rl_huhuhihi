#!/bin/bash

# Robot Arm Deep RL Setup Script for Raspberry Pi 4

echo "=== Robot Arm Deep Reinforcement Learning Setup ==="
echo "Setting up environment for Raspberry Pi 4..."

# Update system packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libjasper-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libfontconfig1-dev \
    libcairo2-dev \
    libgdk-pixbuf2.0-dev \
    libpango1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    python3-pyqt5 \
    python3-h5py \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev

# Enable I2C
echo "Enabling I2C interface..."
sudo raspi-config nonint do_i2c 0

# Add user to i2c group
sudo usermod -a -G i2c $USER

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/python

# Upgrade pip
echo "Upgrading pip..."
.venv/bin/pip install --upgrade pip setuptools wheel

# Install TensorFlow for Raspberry Pi (optimized version)
echo "Installing TensorFlow..."
.venv/bin/pip install tensorflow==2.13.0

# Install other requirements
echo "Installing Python packages..."
.venv/bin/pip install \
    numpy==1.24.3 \
    matplotlib==3.7.2 \
    opencv-python==4.8.0.76 \
    gym==0.26.2 \
    adafruit-circuitpython-pca9685 \
    adafruit-circuitpython-motor \
    RPi.GPIO \
    scipy \
    scikit-learn \
    pandas

# Create directories
echo "Creating project directories..."
mkdir -p models
mkdir -p logs
mkdir -p data
mkdir -p plots

# Set permissions
echo "Setting permissions..."
chmod +x main.py
chmod +x robot_arm_controller.py

# Test I2C connection
echo "Testing I2C connection..."
sudo i2cdetect -y 1

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Connect your PCA9685 to the Raspberry Pi I2C pins"
echo "2. Connect servos to PCA9685 channels 0-3"
echo "3. Activate the virtual environment: source .venv/bin/activate"
echo "4. Test manual control: python main.py --mode manual"
echo "5. Start training: python main.py --mode train --episodes 1000"
echo ""
echo "Hardware connections:"
echo "  Raspberry Pi 4    ->  PCA9685"
echo "  Pin 3 (SDA)       ->  SDA"
echo "  Pin 5 (SCL)       ->  SCL"
echo "  Pin 2 (5V)        ->  VCC"
echo "  Pin 6 (GND)       ->  GND"
echo ""
echo "Connect servos to PCA9685 channels 0, 1, 2, 3"
echo "Ensure adequate power supply for servos (5V, 3A+ recommended)"
