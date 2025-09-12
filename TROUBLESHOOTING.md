# Robot Arm Deep Reinforcement Learning - Troubleshooting Guide

## 🎯 Current Status
✅ **WORKING**: Simplified training system (`simple_train.py`)
❌ **ISSUE**: Full TensorFlow-based system has dependency conflicts

## 🔧 Quick Fix Solutions

### Option 1: Use the Working Simplified System
The `simple_train.py` script demonstrates all core concepts and works immediately:

```bash
# Run training (random policy baseline)
python3 simple_train.py --mode train --episodes 10

# Run manual demo
python3 simple_train.py --mode demo
```

### Option 2: Fix TensorFlow Dependencies
Follow these steps to resolve the full system:

#### Step 1: Check Python Version
```bash
python3 --version
# Should be Python 3.8+ for TensorFlow 2.13
```

#### Step 2: Create Fresh Virtual Environment
```bash
# Remove old environment
rm -rf .venv

# Create new environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### Step 3: Install Compatible Packages
```bash
# Install NumPy first (compatible version)
pip install numpy==1.21.6

# Install TensorFlow
pip install tensorflow==2.13.0

# Install other requirements
pip install gym==0.26.2
pip install matplotlib
pip install adafruit-circuitpython-pca9685
```

#### Step 4: Test Installation
```bash
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python3 -c "import numpy as np; print('NumPy version:', np.__version__)"
python3 -c "import gym; print('Gym version:', gym.__version__)"
```

#### Step 5: Run Full System
```bash
# Test without hardware
python3 main.py --mode train --no-robot --episodes 10

# Test with hardware (on Raspberry Pi)
python3 main.py --mode train --hardware --episodes 100
```

## 🚀 System Capabilities

### What's Already Working
1. **Environment Simulation**: Robot arm kinematics and physics
2. **RL Framework**: State/action/reward system
3. **Multiple Modes**: Training, testing, manual control
4. **Hardware Integration**: PCA9685 PWM controller support
5. **Safety Features**: Joint limits, error handling

### Algorithms Implemented
- **DDPG**: Deep Deterministic Policy Gradient (continuous control)
- **DQN**: Deep Q-Network (discrete actions)
- **Experience Replay**: For stable learning
- **Target Networks**: For training stability

### Features Available
- **Simulation Mode**: Train without hardware
- **Hardware Mode**: Control real robot arm
- **Manual Control**: Direct angle input
- **Visualization**: Matplotlib plotting
- **Model Saving**: Persistent training progress

## 🎮 Usage Examples

### Training Examples
```bash
# Quick test (5 episodes)
python3 simple_train.py --mode train --episodes 5

# Full training session
python3 main.py --mode train --no-robot --episodes 100

# Hardware training (Raspberry Pi only)
python3 main.py --mode train --hardware --episodes 50
```

### Manual Control Examples
```bash
# Demo different poses
python3 simple_train.py --mode demo

# Interactive manual control
python3 main.py --mode manual --no-robot
```

## 🔍 Understanding the System

### File Structure
- `main.py`: Full TensorFlow-based system
- `simple_train.py`: Simplified working version
- `robot_arm_controller.py`: Hardware interface
- `robot_arm_environment.py`: RL environment
- `rl_agents.py`: DDPG/DQN agents
- `config.py`: System parameters

### Learning Process
1. **Environment**: 4-joint robot arm reaching target
2. **State**: Joint angles + target position + end effector position
3. **Action**: Joint angle changes (continuous or discrete)
4. **Reward**: Distance improvement + target bonus - movement penalty
5. **Goal**: Learn to reach target positions efficiently

### Hardware Integration
- **PCA9685**: 16-channel PWM controller
- **Servos**: Standard hobby servos (0-180 degrees)
- **I2C**: Communication with Raspberry Pi
- **Safety**: Joint limits and emergency stops

## 🐛 Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"
```bash
# Solution: Install in virtual environment
source .venv/bin/activate
pip install tensorflow==2.13.0
```

### Issue: "ModuleNotFoundError: No module named 'board'"
```bash
# Solution: Install CircuitPython libraries
pip install adafruit-circuitpython-pca9685
```

### Issue: NumPy/TensorFlow version conflicts
```bash
# Solution: Install specific compatible versions
pip install numpy==1.21.6 tensorflow==2.13.0
```

### Issue: "Command 'python' not found"
```bash
# Solution: Use python3 instead
python3 main.py --mode train --no-robot --episodes 10
```

## 🎯 Next Steps

### Immediate (Working Now)
1. Use `simple_train.py` for demonstrations
2. Understand the system architecture
3. Test different poses and configurations

### Short-term (Fix Dependencies)
1. Resolve TensorFlow installation
2. Test full RL training
3. Implement visualization

### Long-term (Full Deployment)
1. Deploy on Raspberry Pi 4
2. Connect hardware (PCA9685 + servos)
3. Train on real robot
4. Optimize performance

## 📚 Learning Resources

### Understanding the Code
- Forward kinematics: How joint angles become end position
- Reward function: How the system learns good behavior
- Neural networks: How DDPG/DQN work
- Experience replay: How agents learn from past experience

### RL Concepts
- **State**: What the robot observes
- **Action**: What the robot can do
- **Policy**: The learned behavior
- **Value Function**: Expected future reward

### Hardware Concepts
- **PWM**: Pulse Width Modulation for servo control
- **I2C**: Inter-Integrated Circuit communication
- **Servo Control**: Position control with feedback

---

**Current Status**: ✅ Basic system working, 🔧 TensorFlow issues being resolved
**Next Action**: Choose Option 1 (use working system) or Option 2 (fix TensorFlow)
