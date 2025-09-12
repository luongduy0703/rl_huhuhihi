#!/usr/bin/env python3
"""
Test script for robot arm system components
"""

import sys
import numpy as np
import time

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False, False
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False, False
    
    try:
        import gym
        print(f"✓ Gym imported successfully")
    except ImportError as e:
        print(f"✗ Gym import failed: {e}")
        return False, False
    
    # Test hardware-specific imports (may fail on non-Pi systems)
    try:
        import board
        import busio
        from adafruit_pca9685 import PCA9685
        print("✓ Hardware libraries imported successfully")
        hardware_available = True
    except ImportError as e:
        print(f"⚠ Hardware libraries not available: {e}")
        print("  (This is normal on non-Raspberry Pi systems)")
        hardware_available = False
    
    return True, hardware_available

def test_environment_creation():
    """Test environment creation without hardware"""
    print("\nTesting environment creation...")
    
    try:
        from robot_arm_environment import RobotArmEnvironment
        
        # Create environment without hardware controller
        env = RobotArmEnvironment(arm_controller=None, num_joints=4)
        print("✓ Environment created successfully")
        
        # Test reset
        state = env.reset()
        print(f"✓ Environment reset successful, state shape: {state.shape}")
        
        # Test random action
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(f"✓ Environment step successful, reward: {reward:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_agent_creation():
    """Test RL agent creation"""
    print("\nTesting RL agent creation...")
    
    try:
        from rl_agents import DDPGAgent, DQNAgent
        
        state_size = 14  # From environment
        action_size = 4  # Number of joints
        
        # Test DDPG agent
        ddpg_agent = DDPGAgent(state_size=state_size, action_size=action_size)
        print("✓ DDPG agent created successfully")
        
        # Test DQN agent
        dqn_action_size = 5 ** 4  # Discretized actions
        dqn_agent = DQNAgent(state_size=state_size, action_size=dqn_action_size)
        print("✓ DQN agent created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Agent creation test failed: {e}")
        return False

def test_hardware_controller():
    """Test hardware controller (only on Raspberry Pi)"""
    print("\nTesting hardware controller...")
    
    try:
        from robot_arm_controller import RobotArmController
        
        # This will fail gracefully on non-Pi systems
        controller = RobotArmController()
        
        if controller.pca is not None:
            print("✓ Hardware controller initialized successfully")
            
            # Test setting servo angles (safe neutral position)
            controller.set_servo_angle(0, 90)
            time.sleep(0.5)
            controller.cleanup()
            print("✓ Hardware test successful")
            return True
        else:
            print("⚠ Hardware controller created but PCA9685 not available")
            return False
            
    except Exception as e:
        print(f"⚠ Hardware test failed (expected on non-Pi systems): {e}")
        return False

def test_training_simulation():
    """Test a short training simulation"""
    print("\nTesting training simulation...")
    
    try:
        from robot_arm_environment import RobotArmEnvironment
        from rl_agents import DDPGAgent
        
        # Create environment
        env = RobotArmEnvironment(arm_controller=None, num_joints=4)
        
        # Create agent
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        agent = DDPGAgent(state_size=state_size, action_size=action_size)
        
        # Run short training simulation
        state = env.reset()
        total_reward = 0
        
        for step in range(10):  # Short test
            action = agent.act(state, add_noise=True)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Try training step
        if len(agent.memory) >= agent.batch_size:
            agent.replay()
            print("✓ Training step successful")
        
        print(f"✓ Training simulation successful, total reward: {total_reward:.3f}")
        return True
        
    except Exception as e:
        print(f"✗ Training simulation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Robot Arm System Test ===")
    print()
    
    # Test imports
    imports_ok, hardware_available = test_imports()
    if not imports_ok:
        print("\n❌ Critical imports failed. Please install requirements.")
        return False
    
    # Test environment
    env_ok = test_environment_creation()
    if not env_ok:
        print("\n❌ Environment test failed.")
        return False
    
    # Test agents
    agent_ok = test_agent_creation()
    if not agent_ok:
        print("\n❌ Agent creation failed.")
        return False
    
    # Test hardware (optional)
    if hardware_available:
        hardware_ok = test_hardware_controller()
    else:
        hardware_ok = False
        print("⚠ Skipping hardware test (not on Raspberry Pi)")
    
    # Test training simulation
    training_ok = test_training_simulation()
    if not training_ok:
        print("\n❌ Training simulation failed.")
        return False
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"✓ Imports: {'Pass' if imports_ok else 'Fail'}")
    print(f"✓ Environment: {'Pass' if env_ok else 'Fail'}")
    print(f"✓ Agents: {'Pass' if agent_ok else 'Fail'}")
    print(f"✓ Hardware: {'Pass' if hardware_ok else 'Skip/Fail'}")
    print(f"✓ Training: {'Pass' if training_ok else 'Fail'}")
    
    if imports_ok and env_ok and agent_ok and training_ok:
        print("\n🎉 All core tests passed! System is ready.")
        if not hardware_ok:
            print("   (Hardware tests failed - normal on non-Pi systems)")
        return True
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
