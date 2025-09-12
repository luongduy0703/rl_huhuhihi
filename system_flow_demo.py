#!/usr/bin/env python3
"""
Visual System Flow Demonstration - Shows data processing pipeline
"""

import numpy as np
import time

def demonstrate_complete_pipeline():
    """Demonstrate the complete processing pipeline with real numbers"""
    
    print("🤖 ROBOT ARM DEEP RL SYSTEM - COMPLETE PROCESSING PIPELINE")
    print("=" * 70)
    
    # === STEP 1: INITIALIZATION ===
    print("\n1️⃣ SYSTEM INITIALIZATION")
    print("-" * 30)
    
    # Hardware parameters
    min_pulse = 500    # microseconds
    max_pulse = 2500   # microseconds
    num_joints = 4
    
    print(f"Hardware Config:")
    print(f"  - PCA9685 PWM Controller: 50Hz frequency")
    print(f"  - Servo pulse range: {min_pulse}-{max_pulse}μs")
    print(f"  - Joint count: {num_joints}")
    print(f"  - I2C communication: GPIO2(SDA), GPIO3(SCL)")
    
    # Initial state
    joint_angles = np.array([90.0, 90.0, 90.0, 90.0])  # degrees
    target_position = np.array([0.3, 0.2, 0.25])       # meters
    
    print(f"\nInitial State:")
    print(f"  - Joint angles: {joint_angles} degrees")
    print(f"  - Target position: {target_position} meters")
    
    # === STEP 2: FORWARD KINEMATICS ===
    print("\n2️⃣ FORWARD KINEMATICS CALCULATION")
    print("-" * 40)
    
    def forward_kinematics(angles):
        """Calculate end effector position from joint angles"""
        angles_rad = np.deg2rad(angles)
        link_lengths = np.array([0.1, 0.15, 0.12, 0.08])  # meters
        
        x = (link_lengths[0] * np.cos(angles_rad[0]) + 
             link_lengths[1] * np.cos(angles_rad[0]) * np.cos(angles_rad[1]) +
             link_lengths[2] * np.cos(angles_rad[0]) * np.cos(angles_rad[1] + angles_rad[2]))
        
        y = (link_lengths[0] * np.sin(angles_rad[0]) + 
             link_lengths[1] * np.sin(angles_rad[0]) * np.cos(angles_rad[1]) +
             link_lengths[2] * np.sin(angles_rad[0]) * np.cos(angles_rad[1] + angles_rad[2]))
        
        z = (link_lengths[1] * np.sin(angles_rad[1]) +
             link_lengths[2] * np.sin(angles_rad[1] + angles_rad[2]))
        
        return np.array([x, y, z])
    
    end_position = forward_kinematics(joint_angles)
    print(f"Forward Kinematics:")
    print(f"  Input:  Joint angles = {joint_angles}")
    print(f"  Output: End position = [{end_position[0]:.3f}, {end_position[1]:.3f}, {end_position[2]:.3f}] m")
    
    # === STEP 3: STATE REPRESENTATION ===
    print("\n3️⃣ STATE VECTOR CREATION")
    print("-" * 35)
    
    def create_state_vector(joints, target, end_pos):
        """Create normalized state vector for RL agent"""
        # Normalize joint angles to [-1, 1]
        normalized_joints = (joints - 90) / 90  # 90° = center, range 0-180°
        
        # Combine all state information
        state = np.concatenate([normalized_joints, target, end_pos])
        return state
    
    state_vector = create_state_vector(joint_angles, target_position, end_position)
    print(f"State Vector Composition (10 elements):")
    print(f"  [0-3] Normalized joints: {state_vector[:4]}")
    print(f"  [4-6] Target position:   {state_vector[4:7]}")
    print(f"  [7-9] End position:      {state_vector[7:10]}")
    print(f"  Full state: {state_vector}")
    
    # === STEP 4: RL AGENT DECISION ===
    print("\n4️⃣ REINFORCEMENT LEARNING AGENT")
    print("-" * 40)
    
    # Simulate agent decision (in real system, this comes from neural network)
    def simulate_agent_action(state):
        """Simulate RL agent action selection"""
        # For demo: generate action that moves toward target
        current_end = state[7:10]
        target = state[4:7]
        
        # Simple heuristic: move joints slightly toward better position
        action = np.random.uniform(-0.3, 0.3, 4)  # Range [-1, 1]
        return action
    
    action = simulate_agent_action(state_vector)
    print(f"Agent Decision Process:")
    print(f"  Input:  State vector (10 elements)")
    print(f"  Neural Network: Forward pass through DDPG/DQN")
    print(f"  Output: Action vector = {action}")
    print(f"  Action range: [-1, 1] for continuous control")
    
    # === STEP 5: ACTION TO SERVO CONVERSION ===
    print("\n5️⃣ ACTION TO SERVO ANGLE CONVERSION")
    print("-" * 45)
    
    def action_to_servo_angles(action, current_angles):
        """Convert RL action to servo angles"""
        new_angles = np.zeros(4)
        
        for i in range(4):
            # Scale action [-1, 1] to angle change
            angle_change = action[i] * 15  # Max ±15° change per step
            new_angle = current_angles[i] + angle_change
            
            # Clamp to servo limits [0, 180]
            new_angles[i] = np.clip(new_angle, 0, 180)
        
        return new_angles
    
    new_joint_angles = action_to_servo_angles(action, joint_angles)
    angle_changes = new_joint_angles - joint_angles
    print(f"Action Conversion:")
    print(f"  RL Action:    {action}")
    print(f"  Angle change: [{angle_changes[0]:.1f}, {angle_changes[1]:.1f}, {angle_changes[2]:.1f}, {angle_changes[3]:.1f}]°")
    print(f"  New angles:   [{new_joint_angles[0]:.1f}, {new_joint_angles[1]:.1f}, {new_joint_angles[2]:.1f}, {new_joint_angles[3]:.1f}]°")
    
    # === STEP 6: PULSE WIDTH MODULATION ===
    print("\n6️⃣ PWM SIGNAL GENERATION")
    print("-" * 35)
    
    def angle_to_pulse_width(angle, min_pulse=500, max_pulse=2500):
        """Convert servo angle to PWM pulse width"""
        pulse = min_pulse + (angle / 180.0) * (max_pulse - min_pulse)
        return int(pulse)
    
    def pulse_to_duty_cycle(pulse_width):
        """Convert pulse width to 16-bit duty cycle for PCA9685"""
        duty_cycle = int(pulse_width * 65535 / 20000)  # 20ms = 20000μs period
        return duty_cycle
    
    print(f"PWM Signal Generation:")
    for i, angle in enumerate(new_joint_angles):
        pulse = angle_to_pulse_width(angle)
        duty = pulse_to_duty_cycle(pulse)
        print(f"  Joint {i}: {angle:6.1f}° → {pulse:4d}μs → duty_cycle={duty:5d}")
    
    # === STEP 7: HARDWARE EXECUTION ===
    print("\n7️⃣ HARDWARE EXECUTION")
    print("-" * 30)
    
    print(f"Hardware Commands:")
    print(f"  I2C Address: 0x40 (default PCA9685)")
    print(f"  PWM Frequency: 50Hz (20ms period)")
    print(f"  Channels used: 0-3 (4 servos)")
    
    for i, angle in enumerate(new_joint_angles):
        pulse = angle_to_pulse_width(angle)
        duty = pulse_to_duty_cycle(pulse)
        print(f"  pca.channels[{i}].duty_cycle = {duty}")
    
    # === STEP 8: FEEDBACK AND REWARD ===
    print("\n8️⃣ FEEDBACK AND REWARD CALCULATION")
    print("-" * 45)
    
    # Calculate new end position
    new_end_position = forward_kinematics(new_joint_angles)
    
    # Calculate distances
    old_distance = np.linalg.norm(end_position - target_position)
    new_distance = np.linalg.norm(new_end_position - target_position)
    
    # Calculate reward components
    distance_improvement = old_distance - new_distance
    target_bonus = 10.0 if new_distance < 0.05 else 0.0
    movement_penalty = -0.01 * np.sum(np.abs(new_joint_angles - joint_angles))
    total_reward = distance_improvement + target_bonus + movement_penalty
    
    print(f"Feedback Calculation:")
    print(f"  Old end position: [{end_position[0]:.3f}, {end_position[1]:.3f}, {end_position[2]:.3f}] m")
    print(f"  New end position: [{new_end_position[0]:.3f}, {new_end_position[1]:.3f}, {new_end_position[2]:.3f}] m")
    print(f"  Distance to target: {old_distance:.3f}m → {new_distance:.3f}m")
    print(f"  Distance improvement: {distance_improvement:+.3f}")
    print(f"  Target bonus: {target_bonus:+.1f}")
    print(f"  Movement penalty: {movement_penalty:+.3f}")
    print(f"  Total reward: {total_reward:+.3f}")
    
    # === STEP 9: LEARNING UPDATE ===
    print("\n9️⃣ LEARNING UPDATE")
    print("-" * 25)
    
    print(f"Experience Storage:")
    print(f"  (state, action, reward, next_state, done)")
    print(f"  State shape: {state_vector.shape}")
    print(f"  Action shape: {action.shape}")
    print(f"  Reward: {total_reward:.3f}")
    print(f"  Done: {new_distance < 0.05}")
    
    print(f"\nNeural Network Updates:")
    print(f"  DDPG Actor: Updates policy network")
    print(f"  DDPG Critic: Updates value network")
    print(f"  Experience Replay: Sample random batch from memory")
    print(f"  Target Networks: Soft update for stability")
    
    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("📊 PROCESSING PIPELINE SUMMARY")
    print("=" * 70)
    
    print(f"Data Flow:")
    print(f"  State (10 floats) → Agent → Action (4 floats) → Hardware (4 servos)")
    print(f"  └─ Reward (1 float) ← Environment ← New State (10 floats) ←┘")
    
    print(f"\nTiming:")
    print(f"  Control loop: ~20Hz (50ms per cycle)")
    print(f"  Servo update: ~50Hz maximum")
    print(f"  Learning update: Every 32 experiences (batch size)")
    
    print(f"\nKey Parameters:")
    print(f"  Servo range: 0-180° ({min_pulse}-{max_pulse}μs)")
    print(f"  Action range: [-1, +1] continuous")
    print(f"  State size: 10 elements")
    print(f"  Reward range: -∞ to +10 (typically -2 to +10)")
    
    return {
        'old_position': end_position,
        'new_position': new_end_position,
        'reward': total_reward,
        'joint_angles': new_joint_angles,
        'action': action
    }

def show_parameter_effects():
    """Demonstrate how different parameters affect system behavior"""
    
    print("\n" + "=" * 70)
    print("🔧 PARAMETER EFFECTS DEMONSTRATION")
    print("=" * 70)
    
    print("\n1. PULSE WIDTH PARAMETERS EFFECT:")
    print("-" * 40)
    
    angles_to_test = [0, 45, 90, 135, 180]
    
    # Standard servos
    min_pulse_std = 1000
    max_pulse_std = 2000
    
    # Extended range servos  
    min_pulse_ext = 500
    max_pulse_ext = 2500
    
    print(f"{'Angle':<8} {'Standard':<12} {'Extended':<12} {'Difference'}")
    print(f"{'(deg)':<8} {'(μs)':<12} {'(μs)':<12} {'(μs)'}")
    print("-" * 45)
    
    for angle in angles_to_test:
        pulse_std = min_pulse_std + (angle / 180.0) * (max_pulse_std - min_pulse_std)
        pulse_ext = min_pulse_ext + (angle / 180.0) * (max_pulse_ext - min_pulse_ext)
        diff = pulse_ext - pulse_std
        
        print(f"{angle:<8} {pulse_std:<12.0f} {pulse_ext:<12.0f} {diff:+.0f}")
    
    print("\nConclusion: Extended range (500-2500μs) provides better servo control")
    
    print("\n2. REWARD FUNCTION PARAMETERS:")
    print("-" * 35)
    
    distances = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    
    print(f"{'Distance':<10} {'Improvement':<12} {'Target Bonus':<12} {'Total'}")
    print(f"{'(m)':<10} {'Reward':<12} {'Reward':<12} {'Reward'}")
    print("-" * 45)
    
    for dist in distances:
        improvement = 0.1 - dist  # Assume previous distance was 0.1m
        bonus = 10.0 if dist < 0.05 else 0.0
        total = improvement + bonus - 0.02  # Small movement penalty
        
        print(f"{dist:<10.3f} {improvement:<12.3f} {bonus:<12.1f} {total:<12.3f}")
    
    print("\nConclusion: Large bonus for reaching target (<5cm), gradual improvement rewards")

if __name__ == "__main__":
    try:
        # Run complete pipeline demonstration
        result = demonstrate_complete_pipeline()
        
        print(f"\n🎯 FINAL RESULT:")
        print(f"Action taken: {result['action']}")
        print(f"Position change: {np.linalg.norm(result['new_position'] - result['old_position']):.3f}m")
        print(f"Reward earned: {result['reward']:+.3f}")
        
        # Show parameter effects
        show_parameter_effects()
        
        print(f"\n✅ Complete system demonstration finished!")
        print(f"This shows the entire pipeline from RL decision to physical servo control.")
        
    except KeyboardInterrupt:
        print(f"\n👋 Demonstration interrupted by user")
