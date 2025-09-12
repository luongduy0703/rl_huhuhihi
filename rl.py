#!/usr/bin/env python3
"""
Robot Arm Deep Reinforcement Learning - Getting Started Example
This script demonstrates basic usage of the robot arm RL system
"""

import sys
import os
import numpy as np

# Add the RL directory to the Python path
sys.path.append('/home/ducanh/RL')

def basic_example():
    """
    Basic example showing how to use the robot arm system
    """
    print("=== Robot Arm Deep RL - Basic Example ===")
    
    try:
        # Import our custom modules
        from robot_arm_environment import RobotArmEnvironment
        from rl_agents import DDPGAgent
        from robot_arm_controller import RobotArmController
        
        print("✓ All modules imported successfully")
        
        # Option 1: Manual Control Mode
        print("\n1. Manual Control Example:")
        print("   To control servos manually, run:")
        print("   python /home/ducanh/RL/main.py --mode manual")
        print("   This allows you to test servo movements with commands like:")
        print("   - set 0 90    (set servo 0 to 90 degrees)")
        print("   - all 90 90 90 90    (set all servos to 90 degrees)")
        
        # Option 2: Simulation Mode
        print("\n2. Simulation Mode Example:")
        # Create environment without physical robot
        env = RobotArmEnvironment(arm_controller=None, num_joints=4)
        
        # Create DDPG agent
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        agent = DDPGAgent(state_size=state_size, action_size=action_size)
        
        print(f"   Environment created with state size: {state_size}")
        print(f"   Agent created with action size: {action_size}")
        
        # Run a few simulation steps
        state = env.reset()
        print(f"   Initial state shape: {state.shape}")
        
        for step in range(3):
            action = agent.act(state, add_noise=True)
            next_state, reward, done, info = env.step(action)
            print(f"   Step {step + 1}: Reward = {reward:.3f}, Done = {done}")
            
            if done:
                print("   Episode finished early!")
                break
            
            state = next_state
        
        print("   ✓ Simulation test completed successfully")
        
        # Option 3: Training Example
        print("\n3. Training Example:")
        print("   To start training, run:")
        print("   python /home/ducanh/RL/main.py --mode train --agent ddpg --episodes 100 --no-robot")
        print("   (Use --no-robot flag for simulation-only training)")
        
        # Option 4: Hardware Setup
        print("\n4. Hardware Setup:")
        print("   For physical robot, ensure PCA9685 is connected:")
        print("   Raspberry Pi 4    ->  PCA9685")
        print("   Pin 3 (SDA)       ->  SDA")
        print("   Pin 5 (SCL)       ->  SCL") 
        print("   Pin 2 (5V)        ->  VCC")
        print("   Pin 6 (GND)       ->  GND")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please install required packages:")
        print("pip install tensorflow numpy matplotlib gym")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def run_quick_test():
    """
    Run a quick test to verify system is working
    """
    print("\n=== Quick System Test ===")
    
    try:
        # Test system components
        os.system("python /home/ducanh/RL/test_system.py")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    print("Robot Arm Deep Reinforcement Learning System")
    print("=" * 50)
    
    # Run basic example
    success = basic_example()
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 System is ready!")
        print("\nNext steps:")
        print("1. Test manual control: python /home/ducanh/RL/main.py --mode manual")
        print("2. Run system test: python /home/ducanh/RL/test_system.py")
        print("3. Start training: python /home/ducanh/RL/main.py --mode train --no-robot")
        print("4. Check README.md for detailed instructions")
        
        # Ask if user wants to run quick test
        response = input("\nRun quick system test? (y/n): ").strip().lower()
        if response == 'y':
            run_quick_test()
    else:
        print("\n❌ System setup incomplete. Please check error messages above.")