# Robot Arm Configuration
import numpy as np

# Hardware Configuration
NUM_SERVOS = 4
PCA9685_ADDRESS = 0x40
I2C_FREQUENCY = 400000

# Servo Configuration
SERVO_MIN_PULSE = 500    # Minimum pulse width (microseconds)
SERVO_MAX_PULSE = 2500   # Maximum pulse width (microseconds)
SERVO_FREQUENCY = 50     # PWM frequency (Hz)

# Joint angle limits (degrees)
JOINT_LIMITS = [
    (0, 180),    # Base rotation
    (0, 180),    # Shoulder
    (0, 180),    # Elbow
    (0, 180)     # Wrist
]

# Robot Arm Physical Parameters
LINK_LENGTHS = np.array([0.10, 0.15, 0.12, 0.08])  # Link lengths in meters

# Workspace limits (meters)
WORKSPACE_LIMITS = {
    'x': (-0.4, 0.4),
    'y': (-0.4, 0.4),
    'z': (0.0, 0.5)
}

# Training Configuration
TRAINING_CONFIG = {
    'ddpg': {
        'actor_lr': 0.001,
        'critic_lr': 0.002,
        'gamma': 0.99,
        'tau': 0.005,
        'batch_size': 64,
        'memory_size': 100000,
        'noise_std': 0.2
    },
    'dqn': {
        'learning_rate': 0.001,
        'gamma': 0.95,
        'epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 32,
        'memory_size': 50000
    }
}

# Environment Configuration
ENV_CONFIG = {
    'max_steps': 200,
    'target_tolerance': 0.05,  # 5cm tolerance for success
    'reward_weights': {
        'distance': 1.0,
        'target_bonus': 10.0,
        'movement_penalty': 0.1,
        'limit_penalty': 1.0
    }
}

# Safety Configuration
SAFETY_CONFIG = {
    'max_joint_velocity': 30,  # degrees per second
    'emergency_stop_enabled': True,
    'collision_detection': False,  # Implement if sensors available
    'soft_limits_enabled': True
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'robot_arm.log',
    'save_trajectories': True,
    'plot_training': True
}
