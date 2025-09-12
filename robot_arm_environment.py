import numpy as np
from typing import List, Tuple, Dict, Any
import time
import random
from collections import deque

# Try to import TensorFlow and Gym - fall back gracefully if not available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available - some features may be limited")

try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    # Create minimal gym-like interface
    class spaces:
        class Box:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype
            def sample(self):
                return np.random.uniform(self.low, self.high, self.shape)
    
    class gym:
        class Env:
            def __init__(self):
                pass

try:
    import matplotlib.pyplot as plt
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False

class RobotArmEnvironment(gym.Env if GYM_AVAILABLE else object):
    """
    Custom Gym environment for robot arm control
    """
    
    def __init__(self, arm_controller=None, num_joints: int = 4, target_position: np.ndarray = None):
        super(RobotArmEnvironment, self).__init__()
        
        self.arm_controller = arm_controller
        self.num_joints = num_joints
        
        # Action space: continuous angles for each joint (-1 to 1, will be scaled to servo ranges)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_joints,), dtype=np.float32)
        
        # Observation space: current joint angles + target position + end effector position
        # [current_angles(4), target_pos(3), end_effector_pos(3), previous_action(4)]
        obs_dim = num_joints + 3 + 3 + num_joints
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Target position (x, y, z) in workspace
        self.target_position = target_position if target_position is not None else np.array([0.5, 0.0, 0.3])
        
        # Current state
        self.current_joint_angles = np.zeros(num_joints)
        self.previous_action = np.zeros(num_joints)
        
        # Episode parameters
        self.max_steps = 200
        self.current_step = 0
        
        # Reward parameters
        self.previous_distance = float('inf')
        
        # Joint angle limits (in degrees)
        self.joint_limits = [(0, 180), (0, 180), (0, 180), (0, 180)]
        
        # Robot arm parameters (simplified kinematic model)
        self.link_lengths = np.array([0.1, 0.15, 0.12, 0.08])  # meters
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        # Random initial joint angles
        self.current_joint_angles = np.random.uniform(45, 135, self.num_joints)
        self.previous_action = np.zeros(self.num_joints)
        self.current_step = 0
        
        # Set physical robot to initial position
        if self.arm_controller:
            self.arm_controller.set_all_servos(self.current_joint_angles.tolist())
            time.sleep(0.1)  # Allow time for movement
        
        # Calculate initial distance to target
        end_effector_pos = self._forward_kinematics(self.current_joint_angles)
        self.previous_distance = np.linalg.norm(end_effector_pos - self.target_position)
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Scale action from [-1, 1] to servo angles [0, 180]
        scaled_action = self._scale_action(action)
        
        # Apply action to robot arm
        new_joint_angles = np.clip(scaled_action, 
                                 [limit[0] for limit in self.joint_limits],
                                 [limit[1] for limit in self.joint_limits])
        
        # Set physical robot position
        if self.arm_controller:
            self.arm_controller.set_all_servos(new_joint_angles.tolist())
            time.sleep(0.05)  # Small delay for servo movement
        
        self.current_joint_angles = new_joint_angles
        self.previous_action = action
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Additional info
        info = {
            'end_effector_position': self._forward_kinematics(self.current_joint_angles),
            'distance_to_target': np.linalg.norm(self._forward_kinematics(self.current_joint_angles) - self.target_position),
            'joint_angles': self.current_joint_angles.copy()
        }
        
        return self._get_observation(), reward, done, info
    
    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale action from [-1, 1] to joint angle ranges"""
        scaled = np.zeros_like(action)
        for i, (min_angle, max_angle) in enumerate(self.joint_limits):
            # Scale from [-1, 1] to [min_angle, max_angle]
            scaled[i] = min_angle + (action[i] + 1) * (max_angle - min_angle) / 2
        return scaled
    
    def _forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Calculate end effector position using forward kinematics
        Simplified 4-DOF arm model
        """
        angles_rad = np.deg2rad(joint_angles)
        
        # Transform matrices for each joint (simplified)
        x = 0
        y = 0
        z = 0
        
        # Base rotation
        x += self.link_lengths[0] * np.cos(angles_rad[0])
        y += self.link_lengths[0] * np.sin(angles_rad[0])
        
        # Second joint
        x += self.link_lengths[1] * np.cos(angles_rad[0]) * np.cos(angles_rad[1])
        y += self.link_lengths[1] * np.sin(angles_rad[0]) * np.cos(angles_rad[1])
        z += self.link_lengths[1] * np.sin(angles_rad[1])
        
        # Third joint
        x += self.link_lengths[2] * np.cos(angles_rad[0]) * np.cos(angles_rad[1] + angles_rad[2])
        y += self.link_lengths[2] * np.sin(angles_rad[0]) * np.cos(angles_rad[1] + angles_rad[2])
        z += self.link_lengths[2] * np.sin(angles_rad[1] + angles_rad[2])
        
        # End effector
        x += self.link_lengths[3] * np.cos(angles_rad[0]) * np.cos(angles_rad[1] + angles_rad[2] + angles_rad[3])
        y += self.link_lengths[3] * np.sin(angles_rad[0]) * np.cos(angles_rad[1] + angles_rad[2] + angles_rad[3])
        z += self.link_lengths[3] * np.sin(angles_rad[1] + angles_rad[2] + angles_rad[3])
        
        return np.array([x, y, z])
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        # Normalize joint angles to [-1, 1]
        normalized_angles = np.zeros(self.num_joints)
        for i, (min_angle, max_angle) in enumerate(self.joint_limits):
            normalized_angles[i] = 2 * (self.current_joint_angles[i] - min_angle) / (max_angle - min_angle) - 1
        
        end_effector_pos = self._forward_kinematics(self.current_joint_angles)
        
        observation = np.concatenate([
            normalized_angles,           # Current joint angles (normalized)
            self.target_position,        # Target position
            end_effector_pos,           # Current end effector position
            self.previous_action        # Previous action
        ])
        
        return observation.astype(np.float32)
    
    def _calculate_reward(self) -> float:
        """
        Enhanced reward function to address training issues:
        - Less negative rewards overall
        - Stronger distance-based guidance
        - Progressive rewards for getting closer
        """
        end_effector_pos = self._forward_kinematics(self.current_joint_angles)
        current_distance = np.linalg.norm(end_effector_pos - self.target_position)
        
        # 1. DISTANCE IMPROVEMENT REWARD (main component)
        distance_improvement = self.previous_distance - current_distance
        distance_reward = distance_improvement * 10.0  # Amplify improvement signal
        
        # 2. PROXIMITY BONUS (exponential bonus for being close)
        # This provides strong gradient near target
        max_distance = 0.8  # Maximum expected distance in workspace
        proximity_factor = max(0, (max_distance - current_distance) / max_distance)
        proximity_bonus = proximity_factor ** 2 * 2.0  # Quadratic bonus
        
        # 3. MILESTONE REWARDS (progressive rewards)
        milestone_bonus = 0
        if current_distance < 0.30:    # Within 30cm
            milestone_bonus += 1.0
        if current_distance < 0.20:    # Within 20cm
            milestone_bonus += 2.0
        if current_distance < 0.10:    # Within 10cm  
            milestone_bonus += 3.0
        if current_distance < 0.05:    # Within 5cm - SUCCESS!
            milestone_bonus += 15.0
        
        # 4. MOVEMENT EFFICIENCY (reduced penalty)
        movement_penalty = -0.01 * np.sum(np.abs(self.previous_action))  # Much smaller penalty
        
        # 5. SMOOTHNESS REWARD (encourage smooth movements)
        if hasattr(self, 'last_action') and self.last_action is not None:
            action_change = np.sum(np.abs(self.previous_action - self.last_action))
            smoothness_reward = -0.005 * action_change
        else:
            smoothness_reward = 0
        
        # 6. JOINT LIMIT PENALTY (reduced)
        limit_penalty = 0
        for i, angle in enumerate(self.current_joint_angles):
            min_angle, max_angle = self.joint_limits[i]
            if angle <= min_angle + 2 or angle >= max_angle - 2:
                limit_penalty -= 0.5  # Reduced penalty
        
        # 7. TIME STEP PENALTY (small penalty to encourage efficiency)
        time_penalty = -0.01
        
        # TOTAL REWARD CALCULATION
        total_reward = (distance_reward + proximity_bonus + milestone_bonus + 
                       movement_penalty + smoothness_reward + limit_penalty + time_penalty)
        
        # Store for next calculation
        self.last_action = self.previous_action.copy() if self.previous_action is not None else None
        self.previous_distance = current_distance
        
        return total_reward
    
    def _is_episode_done(self) -> bool:
        """Check if episode should terminate"""
        # Episode done if max steps reached
        if self.current_step >= self.max_steps:
            return True
        
        # Episode done if target reached
        end_effector_pos = self._forward_kinematics(self.current_joint_angles)
        distance = np.linalg.norm(end_effector_pos - self.target_position)
        if distance < 0.05:  # 5cm tolerance
            return True
        
        return False
    
    def set_target_position(self, target: np.ndarray):
        """Set new target position"""
        self.target_position = target.copy()
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        if mode == 'human':
            end_effector_pos = self._forward_kinematics(self.current_joint_angles)
            distance = np.linalg.norm(end_effector_pos - self.target_position)
            print(f"Step: {self.current_step}, Distance to target: {distance:.4f}")
            print(f"Joint angles: {self.current_joint_angles}")
            print(f"End effector: {end_effector_pos}")
            print(f"Target: {self.target_position}")
            print("-" * 50)
