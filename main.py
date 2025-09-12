#!/usr/bin/env python3
"""
Main training script for robot arm deep reinforcement learning
Note: Use 'python3' command instead of 'python' on this system
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from typing import Optional

# Import our custom modules
from robot_arm_controller import RobotArmController
from robot_arm_environment import RobotArmEnvironment
from rl_agents import DDPGAgent, DQNAgent

class RobotArmTrainer:
    """
    Main trainer class for robot arm RL
    """
    
    def __init__(self, 
                 use_physical_robot: bool = True,
                 num_joints: int = 4,
                 agent_type: str = 'ddpg',
                 model_save_path: str = 'models/robot_arm'):
        """
        Initialize trainer
        
        Args:
            use_physical_robot: Whether to use physical robot or simulation only
            num_joints: Number of robot arm joints
            agent_type: Type of RL agent ('ddpg' or 'dqn')
            model_save_path: Path to save trained models
        """
        self.use_physical_robot = use_physical_robot
        self.num_joints = num_joints
        self.agent_type = agent_type.lower()
        self.model_save_path = model_save_path
        
        # Create models directory
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Initialize robot controller
        self.robot_controller = None
        if use_physical_robot:
            try:
                self.robot_controller = RobotArmController(num_servos=num_joints)
                print("Physical robot controller initialized")
            except Exception as e:
                print(f"Failed to initialize robot controller: {e}")
                print("Running in simulation mode only")
                self.use_physical_robot = False
        
        # Initialize environment
        target_position = np.array([0.3, 0.2, 0.25])  # Target position in workspace
        self.env = RobotArmEnvironment(
            arm_controller=self.robot_controller,
            num_joints=num_joints,
            target_position=target_position
        )
        
        # Initialize RL agent
        state_size = self.env.observation_space.shape[0]
        
        if self.agent_type == 'ddpg':
            action_size = self.env.action_space.shape[0]
            self.agent = DDPGAgent(
                state_size=state_size,
                action_size=action_size,
                actor_lr=0.001,
                critic_lr=0.002,
                gamma=0.99,
                batch_size=64,
                memory_size=100000
            )
        else:  # DQN with discretized actions
            # Discretize continuous action space
            self.action_discretization = 5  # 5 levels per joint
            action_size = self.action_discretization ** num_joints
            self.agent = DQNAgent(
                state_size=state_size,
                action_size=action_size,
                learning_rate=0.001,
                gamma=0.95,
                batch_size=32,
                memory_size=50000
            )
        
        print(f"Initialized {self.agent_type.upper()} agent")
        print(f"State size: {state_size}")
        print(f"Action size: {action_size if self.agent_type == 'ddpg' else action_size}")
    
    def discretize_action(self, action_index: int) -> np.ndarray:
        """Convert discrete action index to continuous action for DQN"""
        # Convert single index to multi-dimensional action
        actions = []
        remaining = action_index
        
        for _ in range(self.num_joints):
            actions.append(remaining % self.action_discretization)
            remaining //= self.action_discretization
        
        # Convert to continuous values [-1, 1]
        continuous_actions = []
        for discrete_val in actions:
            continuous_val = -1 + 2 * discrete_val / (self.action_discretization - 1)
            continuous_actions.append(continuous_val)
        
        return np.array(continuous_actions)
    
    def train(self, 
              episodes: int = 1000,
              max_steps_per_episode: int = 200,
              save_interval: int = 100,
              target_update_interval: int = 10,
              render_interval: int = 50):
        """
        Train the RL agent
        
        Args:
            episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            save_interval: Save model every N episodes
            target_update_interval: Update target network every N episodes
            render_interval: Print progress every N episodes
        """
        print(f"Starting training for {episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            step_count = 0
            
            for step in range(max_steps_per_episode):
                # Select action
                if self.agent_type == 'ddpg':
                    action = self.agent.act(state, add_noise=True)
                else:  # DQN
                    action_index = self.agent.act(state)
                    action = self.discretize_action(action_index)
                
                # Execute action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                if self.agent_type == 'ddpg':
                    self.agent.remember(state, action, reward, next_state, done)
                else:  # DQN
                    self.agent.remember(state, action_index, reward, next_state, done)
                
                # Train agent
                if self.agent_type == 'ddpg':
                    actor_loss, critic_loss = self.agent.replay()
                else:
                    loss = self.agent.replay()
                
                state = next_state
                episode_reward += reward
                step_count += 1
                
                if done:
                    break
            
            # Update target network for DQN
            if self.agent_type == 'dqn' and episode % target_update_interval == 0:
                self.agent.update_target_network()
            
            # Store metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            self.agent.reward_history.append(episode_reward)
            
            # Print progress
            if episode % render_interval == 0:
                avg_reward = np.mean(episode_rewards[-render_interval:])
                avg_length = np.mean(episode_lengths[-render_interval:])
                
                print(f"Episode {episode}/{episodes}")
                print(f"  Average Reward: {avg_reward:.2f}")
                print(f"  Average Length: {avg_length:.1f}")
                
                if self.agent_type == 'ddpg':
                    if len(self.agent.actor_loss_history) > 0:
                        print(f"  Actor Loss: {self.agent.actor_loss_history[-1]:.4f}")
                        print(f"  Critic Loss: {self.agent.critic_loss_history[-1]:.4f}")
                else:
                    print(f"  Exploration Rate: {self.agent.epsilon:.3f}")
                
                print(f"  Distance to Target: {info.get('distance_to_target', 'N/A'):.4f}")
                print("-" * 50)
            
            # Save model
            if episode > 0 and episode % save_interval == 0:
                self.agent.save_model(f"{self.model_save_path}_episode_{episode}")
                print(f"Model saved at episode {episode}")
        
        # Final save
        self.agent.save_model(self.model_save_path)
        print("Training completed!")
        
        return episode_rewards, episode_lengths
    
    def test(self, 
             num_episodes: int = 10,
             render: bool = True,
             model_path: Optional[str] = None):
        """
        Test the trained agent
        
        Args:
            num_episodes: Number of test episodes
            render: Whether to render environment
            model_path: Path to load model from
        """
        if model_path:
            self.agent.load_model(model_path)
        
        print(f"Testing agent for {num_episodes} episodes...")
        
        test_rewards = []
        success_count = 0
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            step_count = 0
            
            for step in range(200):  # Max steps for test
                # Select action (no exploration)
                if self.agent_type == 'ddpg':
                    action = self.agent.act(state, add_noise=False)
                else:  # DQN
                    # Use greedy policy for testing
                    old_epsilon = self.agent.epsilon
                    self.agent.epsilon = 0.0
                    action_index = self.agent.act(state)
                    action = self.discretize_action(action_index)
                    self.agent.epsilon = old_epsilon
                
                next_state, reward, done, info = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                step_count += 1
                
                if render:
                    self.env.render()
                    time.sleep(0.1)  # Small delay for visualization
                
                if done:
                    if info.get('distance_to_target', float('inf')) < 0.05:
                        success_count += 1
                    break
            
            test_rewards.append(episode_reward)
            print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step_count}")
        
        avg_reward = np.mean(test_rewards)
        success_rate = success_count / num_episodes
        
        print(f"\nTest Results:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Success Rate: {success_rate:.2%}")
        
        return test_rewards, success_rate
    
    def plot_training_results(self, episode_rewards: list, episode_lengths: list):
        """Plot training results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        ax1.plot(episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Moving average of rewards
        window_size = min(100, len(episode_rewards) // 10)
        if window_size > 1:
            moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(range(window_size-1, len(episode_rewards)), moving_avg, 'r-', alpha=0.7, label='Moving Average')
            ax1.legend()
        
        # Episode lengths
        ax2.plot(episode_lengths)
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True)
        
        # Loss history
        if self.agent_type == 'ddpg' and len(self.agent.actor_loss_history) > 0:
            ax3.plot(self.agent.actor_loss_history, label='Actor Loss')
            ax3.plot(self.agent.critic_loss_history, label='Critic Loss')
            ax3.set_title('Training Losses')
            ax3.legend()
        elif hasattr(self.agent, 'loss_history') and len(self.agent.loss_history) > 0:
            ax3.plot(self.agent.loss_history)
            ax3.set_title('Training Loss')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Loss')
        ax3.grid(True)
        
        # Exploration rate (for DQN)
        if self.agent_type == 'dqn':
            episodes = len(episode_rewards)
            epsilon_values = [self.agent.epsilon_min + (1.0 - self.agent.epsilon_min) * 
                            (self.agent.epsilon_decay ** ep) for ep in range(episodes)]
            ax4.plot(epsilon_values)
            ax4.set_title('Exploration Rate (Epsilon)')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Epsilon')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.robot_controller:
            self.robot_controller.cleanup()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Robot Arm Deep Reinforcement Learning')
    parser.add_argument('--mode', choices=['train', 'test', 'manual'], default='train',
                       help='Operation mode')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--agent', choices=['ddpg', 'dqn'], default='ddpg',
                       help='RL agent type')
    parser.add_argument('--no-robot', action='store_true',
                       help='Run in simulation only (no physical robot)')
    parser.add_argument('--model-path', type=str, default='models/robot_arm',
                       help='Path to save/load model')
    parser.add_argument('--test-episodes', type=int, default=10,
                       help='Number of test episodes')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = RobotArmTrainer(
            use_physical_robot=not args.no_robot,
            agent_type=args.agent,
            model_save_path=args.model_path
        )
        
        if args.mode == 'train':
            # Train the agent
            episode_rewards, episode_lengths = trainer.train(episodes=args.episodes)
            
            # Plot results
            trainer.plot_training_results(episode_rewards, episode_lengths)
            
        elif args.mode == 'test':
            # Test the agent
            trainer.test(num_episodes=args.test_episodes, model_path=args.model_path)
            
        elif args.mode == 'manual':
            # Manual control mode
            from robot_arm_controller import manual_control_interface
            manual_control_interface()
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'trainer' in locals():
            trainer.cleanup()

if __name__ == "__main__":
    main()
