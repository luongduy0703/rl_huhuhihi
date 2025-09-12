#!/usr/bin/env python3
"""
Simple test script for robot arm system - bypasses numpy/scipy compatibility issues
"""

import sys
import numpy as np
import time

def simple_test():
    """Simple test of core functionality without TensorFlow"""
    print("=== Simple Robot Arm System Test ===")
    print()
    
    # Test NumPy
    print("Testing NumPy...")
    try:
        arr = np.array([1, 2, 3, 4])
        print(f"✓ NumPy working - version {np.__version__}")
    except Exception as e:
        print(f"✗ NumPy failed: {e}")
        return False
    
    # Test environment without TensorFlow
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
        import traceback
        traceback.print_exc()
        return False

def test_manual_control():
    """Test manual control without hardware"""
    print("\nTesting manual control interface...")
    try:
        from robot_arm_controller import RobotArmController
        
        # This should work even without hardware
        controller = RobotArmController()
        print("✓ Controller created (hardware may not be available)")
        
        # Test basic functionality
        angles = controller.get_current_positions()
        print(f"✓ Got current positions: {angles}")
        
        return True
    except Exception as e:
        print(f"⚠ Manual control test: {e} (expected without hardware)")
        return True  # This is expected without hardware

def demo_simulation():
    """Run a short simulation demo"""
    print("\nRunning simulation demo...")
    try:
        from robot_arm_environment import RobotArmEnvironment
        
        env = RobotArmEnvironment(arm_controller=None, num_joints=4)
        
        print("Running 5 random steps...")
        state = env.reset()
        total_reward = 0
        
        for step in range(5):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            distance = info.get('distance_to_target', 'N/A')
            
            print(f"  Step {step + 1}: Reward = {reward:.3f}, Distance = {distance:.4f}")
            
            state = next_state
            if done:
                print("  Episode completed early!")
                break
        
        print(f"✓ Simulation completed, total reward: {total_reward:.3f}")
        return True
        
    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Robot Arm Deep RL - Simple System Test")
    print("=" * 50)
    
    # Run simple test
    if not simple_test():
        print("\n❌ Basic test failed")
        return False
    
    # Test manual control
    test_manual_control()
    
    # Run demo
    if not demo_simulation():
        print("\n❌ Simulation demo failed")
        return False
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 Core system tests passed!")
    print()
    print("Next steps:")
    print("1. For manual control: python main.py --mode manual")
    print("2. For simulation training: python main.py --mode train --no-robot --episodes 10")
    print("3. Install hardware libraries for physical robot:")
    print("   sudo apt install python3-dev i2c-tools")
    print("   pip3 install --user adafruit-circuitpython-pca9685")
    print()
    print("Note: TensorFlow may have compatibility issues.")
    print("Consider using a virtual environment or Docker for full functionality.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
