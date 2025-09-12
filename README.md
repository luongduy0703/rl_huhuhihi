# 🤖 Robot## 📁 Project Structure

```
robot-arm-rl-complete-project/
├── 📜 README.md                    # Main documentation
├── 📋 requirements.txt             # Python dependencies
├── 🍓 PI4_INSTALLATION_GUIDE.md    # Raspberry Pi 4 setup guide
├── 📖 DETAILED_DOCUMENTATION.md    # Complete technical docs
├── 🎯 enhanced_trainer.py          # Main training system
├── ⚙️ advanced_config.py           # Configuration & benchmarks
├── 🤖 rl_agents.py                 # DDPG/DQN implementations
├── 🦾 robot_arm_controller.py      # Hardware control (PCA9685)
├── 🌍 robot_arm_environment.py     # RL environment
├── 📊 plots/                       # Training visualizations
├── 💾 models/                      # Saved neural networks
└── 📈 metrics/                     # Training metrics
```

**That's it!** Just **8 core files** + folders. Clean and focused.cement Learning System

**Enhanced Version** - Clean, optimized codebase with proven performance improvements.

A deep reinforcement learning system for controlling robot arms using TensorFlow on Raspberry Pi 4. **Major achievements**:
- ✅ **Rewards**: +68 to +610 (from -30 to -15)
- ✅ **Distance**: 16-19cm from target (from 32-40cm)  
- ✅ **Stability**: Consistent learning with enhanced reward function
- ✅ **Analytics**: Comprehensive metrics and visualization tools

## 📁 Project Structure

```
robot-arm-rl-complete-project/
├── 📜 README.md                    # This guide
├── 📋 requirements.txt             # Python dependencies
├── 🍓 PI4_INSTALLATION_GUIDE.md    # Raspberry Pi 4 setup
├── � DETAILED_DOCUMENTATION.md    # Technical documentation
├── 🎯 enhanced_trainer.py          # Main training script
├── ⚙️ advanced_config.py           # Configuration system
├── 🤖 rl_agents.py                 # DDPG and DQN agents  
├── 🦾 robot_arm_controller.py      # Hardware control
├── 🌍 robot_arm_environment.py     # RL environment
├── 📊 plots/                       # Training visualizations
├── 💾 models/                      # Saved neural networks
└── 📈 metrics/                     # Training metrics
```

## 🌟 Key Features

- **🚀 Enhanced Training**: Optimized reward function and learning stability
- **📊 Comprehensive Metrics**: Success rates, efficiency tracking, real-time visualization
- **🔧 Advanced Configuration**: Hyperparameter tuning and benchmarks
- **🎯 Success Benchmarks**: Clear performance targets and evaluation
- **️ Hardware Integration**: PCA9685 PWM controller with full servo range
- **🖥️ Simulation Mode**: Complete functionality without physical hardware
- **💾 Model Persistence**: Save/load trained models with metrics

## Hardware Requirements

- Raspberry Pi 4 (4GB+ recommended)
- PCA9685 16-Channel PWM Driver
- 4x Servo motors (e.g., SG90, MG996R)
- Adequate power supply (5V, 3A+ for servos)
- Jumper wires and breadboard

## Hardware Connections

```
Raspberry Pi 4    ->  PCA9685
Pin 3 (SDA)       ->  SDA
Pin 5 (SCL)       ->  SCL
Pin 2 (5V)        ->  VCC
Pin 6 (GND)       ->  GND
```

Connect servos to PCA9685 channels 0, 1, 2, 3 (Base, Shoulder, Elbow, Wrist)

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/do010303/robot-arm-rl-complete-project.git
   cd robot-arm-rl-complete-project
   ```

2. **Set up virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Training

```bash
# Basic training (simulation mode)
python3 enhanced_trainer.py --mode train --no-robot --episodes 25

# Training with hardware
python3 enhanced_trainer.py --mode train --episodes 100

# Quick test
python3 enhanced_trainer.py --mode train --no-robot --episodes 5
```

### Testing Trained Model

```bash
# Test saved model
python3 enhanced_trainer.py --mode test --episodes 10 --load-model

# Evaluate performance
python3 enhanced_trainer.py --mode eval --episodes 50
```

## ⚙️ Configuration

Customize training in `advanced_config.py`:

- **Hardware settings**: Servo limits, PWM frequencies
- **Training parameters**: Learning rates, batch sizes, network architectures  
- **Environment settings**: Reward functions, episode lengths
- **Performance benchmarks**: Success thresholds and evaluation metrics

## Robot Arm Kinematics

The system uses a simplified 4-DOF (Degrees of Freedom) robot arm model:

1. **Base Joint**: Rotation around vertical axis (0-180°)
2. **Shoulder Joint**: Forward/backward arm movement (0-180°)
3. **Elbow Joint**: Elbow bend (0-180°)
4. **Wrist Joint**: Wrist rotation (0-180°)

## Deep Reinforcement Learning

### DDPG (Deep Deterministic Policy Gradient)
- **Best for**: Continuous control tasks
- **Action Space**: Continuous joint angles
- **Networks**: Actor-Critic architecture
- **Exploration**: Gaussian noise

### DQN (Deep Q-Network)
- **Best for**: Discrete control tasks
- **Action Space**: Discretized joint movements
- **Networks**: Value function approximation
- **Exploration**: ε-greedy policy

## Training Process

1. **Environment Reset**: Robot arm starts at random position
2. **Action Selection**: Agent selects joint angles
3. **Environment Step**: Servo motors move to new positions
4. **Reward Calculation**: Based on distance to target
5. **Experience Storage**: Store state-action-reward transitions
6. **Network Training**: Update neural networks
7. **Target Updates**: Periodic target network updates

## Reward Function

The reward function encourages:
- **Reaching the target**: Large bonus for success
- **Moving towards target**: Positive reward for distance reduction
- **Smooth movements**: Penalty for large joint movements
- **Staying within limits**: Penalty for approaching joint limits

## Safety Features

- **Joint Limits**: Software limits prevent servo damage
- **Smooth Movements**: Gradual servo transitions
- **Emergency Stop**: Manual interrupt capability
- **Power Management**: Controlled servo power

## 🔧 Troubleshooting

### TensorFlow Issues
```bash
# For Pi 4 - install optimized version
pip install --extra-index-url https://www.piwheels.org/simple/ tensorflow==2.15.1
```

### Environment Issues
```bash
# Check virtual environment
python3 -c "import tensorflow as tf; print('TF version:', tf.__version__)"

# Test system
python3 enhanced_trainer.py --mode train --no-robot --episodes 3
```

### Hardware Issues
```bash
# Check I2C (Pi 4)
sudo i2cdetect -y 1

# Verify connections and power supply
```

For detailed troubleshooting, see `DETAILED_DOCUMENTATION.md`

## 📚 Documentation

- **📖 DETAILED_DOCUMENTATION.md**: Complete technical documentation
- **🍓 PI4_INSTALLATION_GUIDE.md**: Raspberry Pi 4 specific setup
- **📜 README.md**: This overview guide

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Test thoroughly (especially on Pi 4)
4. Submit pull request

## 📄 License

Open source project. See repository for license details.

**Happy robot learning!** 🤖✨
