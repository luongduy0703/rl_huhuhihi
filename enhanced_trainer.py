#!/usr/bin/env python3
"""
Enhanced training script with comprehensive metrics tracking and visualization
"""

import os
import sys
import numpy as np

# Set matplotlib to non-interactive backend for faster execution
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import time
import argparse
from typing import List, Dict, Optional
from collections import deque

# Import our custom modules
from robot_arm_controller import RobotArmController
from robot_arm_environment import RobotArmEnvironment
from rl_agents import DDPGAgent, DQNAgent

class EnhancedMetricsTracker:
    """Enhanced metrics tracking with real-time visualization"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        
        # Metrics storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.distances_to_target = []
        self.success_rates = []
        self.exploration_rates = []
        
        # Additional tracking for enhanced analysis
        self.improvement_rates = []  # Rate of distance improvement
        self.efficiency_scores = []  # Reward per step
        self.learning_stability = []  # Loss variance tracking
        
        # Moving averages - optimized with deque
        self.reward_window = deque(maxlen=window_size)
        self.length_window = deque(maxlen=window_size)
        self.distance_window = deque(maxlen=window_size)
        self.success_window = deque(maxlen=window_size)
        
        # Pre-computed moving averages for plotting (updated incrementally)
        self.reward_moving_avg = []
        self.distance_moving_avg = []
        
        # Success tracking
        self.success_threshold = 0.05  # 5cm success threshold
        self.consecutive_successes = 0
        self.best_distance = float('inf')
        self.total_successes = 0
        self.best_episode_reward = float('-inf')
        
    def update(self, episode: int, reward: float, length: int, 
               distance: float, actor_loss: float = None, 
               critic_loss: float = None, epsilon: float = None):
        """Update all metrics"""
        
        # Store raw values
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.distances_to_target.append(distance)
        
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)
        if critic_loss is not None:
            self.critic_losses.append(critic_loss)
        if epsilon is not None:
            self.exploration_rates.append(epsilon)
            
        # Update moving windows
        self.reward_window.append(reward)
        self.length_window.append(length)
        self.distance_window.append(distance)
        
        # Update pre-computed moving averages for faster plotting
        current_reward_avg = np.mean(self.reward_window)
        current_distance_avg = np.mean(self.distance_window)
        self.reward_moving_avg.append(current_reward_avg)
        self.distance_moving_avg.append(current_distance_avg)
        
        # Success tracking
        is_success = distance < self.success_threshold
        self.success_window.append(1.0 if is_success else 0.0)
        
        if is_success:
            self.consecutive_successes += 1
            self.total_successes += 1
            if distance < self.best_distance:
                self.best_distance = distance
        else:
            self.consecutive_successes = 0
            
        # Calculate success rate
        current_success_rate = np.mean(self.success_window) if self.success_window else 0.0
        self.success_rates.append(current_success_rate)
        
        # Additional metrics
        if len(self.episode_rewards) > 1:
            # Improvement rate (distance getting better)
            if len(self.distances_to_target) > 1:
                distance_improvement = self.distances_to_target[-2] - distance
                self.improvement_rates.append(distance_improvement)
            
            # Efficiency score (reward per step)
            efficiency = reward / max(length, 1)
            self.efficiency_scores.append(efficiency)
            
            # Track best episode reward
            if reward > self.best_episode_reward:
                self.best_episode_reward = reward
    
    def get_current_stats(self, compute_derived=True) -> Dict:
        """Get current statistics with enhanced metrics"""
        stats = {
            'avg_reward': np.mean(self.reward_window) if self.reward_window else 0,
            'avg_length': np.mean(self.length_window) if self.length_window else 0,
            'avg_distance': np.mean(self.distance_window) if self.distance_window else 0,
            'success_rate': np.mean(self.success_window) if self.success_window else 0,
            'consecutive_successes': self.consecutive_successes,
            'total_successes': self.total_successes,
            'best_distance': self.best_distance if self.best_distance != float('inf') else 0,
            'best_episode_reward': self.best_episode_reward if self.best_episode_reward != float('-inf') else 0,
            'actor_loss': self.actor_losses[-1] if self.actor_losses else 0,
            'critic_loss': self.critic_losses[-1] if self.critic_losses else 0,
            'exploration_rate': self.exploration_rates[-1] if self.exploration_rates else 0
        }
        
        # Add derived metrics only when needed (for detailed reports)
        if compute_derived:
            if self.improvement_rates:
                stats['avg_improvement'] = np.mean(self.improvement_rates[-self.window_size:])
            else:
                stats['avg_improvement'] = 0
                
            if self.efficiency_scores:
                stats['efficiency'] = np.mean(self.efficiency_scores[-self.window_size:])
            else:
                stats['efficiency'] = 0
        else:
            # Use cached values for fast access
            stats['avg_improvement'] = 0
            stats['efficiency'] = 0
            
        return stats
    
    def plot_training_progress(self, save_path: str = None):
        """Create comprehensive training progress plots"""
        
        if len(self.episode_rewards) < 2:
            print("Not enough data to plot")
            return
            
        episodes = range(len(self.episode_rewards))
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Robot Arm RL Training Progress', fontsize=16, fontweight='bold')
        
        # 1. Rewards over time
        axes[0, 0].plot(episodes, self.episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
        if len(self.reward_moving_avg) > 0:
            axes[0, 0].plot(episodes, self.reward_moving_avg, color='red', linewidth=2, label=f'Moving Avg ({self.window_size})')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. Distance to Target
        axes[0, 1].plot(episodes, self.distances_to_target, alpha=0.3, color='green', label='Distance')
        if len(self.distance_moving_avg) > 0:
            axes[0, 1].plot(episodes, self.distance_moving_avg, color='red', linewidth=2, label=f'Moving Avg ({self.window_size})')
        axes[0, 1].axhline(y=self.success_threshold, color='orange', linestyle='--', label='Success Threshold')
        axes[0, 1].set_title('Distance to Target')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Distance (m)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 3. Success Rate
        axes[0, 2].plot(episodes, [r * 100 for r in self.success_rates], color='purple', linewidth=2)
        axes[0, 2].set_title('Success Rate')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Success Rate (%)')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 100)
        
        # 4. Actor Loss (if available)
        if self.actor_losses:
            axes[1, 0].plot(range(len(self.actor_losses)), self.actor_losses, color='orange', linewidth=2)
            axes[1, 0].set_title('Actor Loss')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Actor Loss Data', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Actor Loss (N/A)')
        
        # 5. Critic Loss (if available)
        if self.critic_losses:
            axes[1, 1].plot(range(len(self.critic_losses)), self.critic_losses, color='red', linewidth=2)
            axes[1, 1].set_title('Critic Loss')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Critic Loss Data', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Critic Loss (N/A)')
        
        # 6. Episode Lengths
        axes[1, 2].plot(episodes, self.episode_lengths, alpha=0.6, color='brown', linewidth=1)
        axes[1, 2].set_title('Episode Lengths')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Steps')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training progress plot saved to: {save_path}")
        
        # Close the figure to free memory
        plt.close(fig)
        
    def save_metrics(self, save_path: str):
        """Save metrics to file"""
        metrics_data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'distances_to_target': self.distances_to_target,
            'success_rates': self.success_rates,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'exploration_rates': self.exploration_rates,
            'best_distance': self.best_distance,
            'success_threshold': self.success_threshold
        }
        
        np.save(save_path, metrics_data)
        print(f"Metrics saved to: {save_path}")

class EnhancedRobotArmTrainer:
    """Enhanced trainer with comprehensive metrics and analysis"""
    
    def __init__(self, 
                 use_physical_robot: bool = True,
                 num_joints: int = 4,
                 agent_type: str = 'ddpg',
                 model_save_path: str = 'models/robot_arm'):
        
        self.use_physical_robot = use_physical_robot
        self.num_joints = num_joints
        self.agent_type = agent_type.lower()
        self.model_save_path = model_save_path
        
        # Create directories
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        os.makedirs('metrics', exist_ok=True)
        
        # Initialize components
        self.robot_controller = None
        if use_physical_robot:
            try:
                self.robot_controller = RobotArmController()
                print("✅ Physical robot controller initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize physical robot: {e}")
                print("🔄 Continuing in simulation mode")
                self.use_physical_robot = False
        
        # Initialize environment
        self.env = RobotArmEnvironment(
            arm_controller=self.robot_controller,
            num_joints=num_joints
        )
        
        # Initialize agent
        # Get state size by resetting environment and checking observation shape
        initial_state = self.env.reset()
        state_size = initial_state.shape[0]
        action_size = num_joints
        
        if agent_type == 'ddpg':
            self.agent = DDPGAgent(state_size, action_size)
        else:
            action_space_size = 5 ** num_joints  # 5 discrete actions per joint
            self.agent = DQNAgent(state_size, action_space_size)
        
        # Initialize metrics tracker
        self.metrics = EnhancedMetricsTracker()
        
        print(f"🤖 Initialized {agent_type.upper()} agent")
        print(f"📊 State size: {state_size}")
        print(f"🎯 Action size: {action_size}")
        
    def train(self, 
              episodes: int = 1000,
              max_steps_per_episode: int = 200,
              save_interval: int = 100,
              render_interval: int = 10,
              fast_mode: bool = False):
        """Enhanced training with comprehensive metrics"""
        
        print(f"\n🚀 Starting enhanced training for {episodes} episodes...")
        if fast_mode:
            print("⚡ Fast mode enabled - reduced output for maximum speed")
        print(f"📈 Success threshold: {self.metrics.success_threshold:.3f}m")
        print("-" * 60)
        
        start_time = time.time()
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            step_count = 0
            final_distance = float('inf')
            
            for step in range(max_steps_per_episode):
                # Select action
                if self.agent_type == 'ddpg':
                    action = self.agent.act(state, add_noise=True)
                else:  # DQN
                    action_index = self.agent.act(state)
                    action = self._discretize_action(action_index)
                
                # Execute action
                next_state, reward, done, info = self.env.step(action)
                final_distance = info.get('distance_to_target', final_distance)
                
                # Store experience
                if self.agent_type == 'ddpg':
                    self.agent.remember(state, action, reward, next_state, done)
                    actor_loss, critic_loss = self.agent.replay()
                else:
                    self.agent.remember(state, action_index, reward, next_state, done)
                    loss = self.agent.replay()
                    actor_loss, critic_loss = None, loss
                
                state = next_state
                episode_reward += reward
                step_count += 1
                
                if done:
                    break
            
            # Update metrics
            self.metrics.update(
                episode=episode,
                reward=episode_reward,
                length=step_count,
                distance=final_distance,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                epsilon=getattr(self.agent, 'epsilon', None)
            )
            
            # Print progress based on mode
            if fast_mode:
                # Fast mode: only print every 10 episodes
                if episode % 10 == 0 or episode == episodes - 1:
                    stats = self.metrics.get_current_stats(compute_derived=False)
                    print(f"Ep {episode:3d}/{episodes} | R:{episode_reward:6.1f} | d:{final_distance:.3f}m | "
                          f"AvgR:{stats['avg_reward']:6.1f} | Avgd:{stats['avg_distance']:.3f}m | "
                          f"SR:{stats['success_rate']*100:4.1f}% | Steps:{step_count:3d}")
            else:
                # Normal mode: print every episode with details
                stats = self.metrics.get_current_stats(compute_derived=False)
                
                # Minimized single-line progress for every episode
                print(f"Ep {episode:3d}/{episodes} | R:{episode_reward:6.1f} | d:{final_distance:.3f}m | "
                      f"AvgR:{stats['avg_reward']:6.1f} | Avgd:{stats['avg_distance']:.3f}m | "
                      f"SR:{stats['success_rate']*100:4.1f}% | Steps:{step_count:3d}")
                
                # Detailed progress every render_interval episodes (skip episode 0) - use full stats
                if episode % render_interval == 0 and episode > 0:
                    stats = self.metrics.get_current_stats(compute_derived=True)  # Full computation for detailed report
                    elapsed_time = time.time() - start_time  # Only calculate when needed
                    print(f"\n{'='*60}")
                    print(f"📊 DETAILED PROGRESS - Episode {episode:4d}/{episodes}")
                    print(f"{'='*60}")
                    print(f"  🏆 Avg Reward: {stats['avg_reward']:8.2f} (Best: {stats['best_episode_reward']:6.1f})")
                    print(f"  📏 Avg Distance: {stats['avg_distance']:6.4f}m (Best: {stats['best_distance']:6.4f}m)")
                    print(f"  ✅ Success Rate: {stats['success_rate']*100:5.1f}% (Total: {stats['total_successes']})")
                    print(f"  🔥 Consecutive: {stats['consecutive_successes']:3d}")
                    print(f"  📈 Improvement: {stats['avg_improvement']:+6.4f}m/ep")
                    print(f"  ⚡ Efficiency: {stats['efficiency']:6.2f} reward/step")
                    
                    if self.agent_type == 'ddpg':
                        print(f"  🎭 Actor Loss: {stats['actor_loss']:8.4f}")
                        print(f"  🧠 Critic Loss: {stats['critic_loss']:8.4f}")
                    else:
                        print(f"  🔍 Exploration: {stats['exploration_rate']:6.3f}")
                    
                    print(f"  ⏱️ Time: {elapsed_time/60:.1f}min")
                    print(f"{'='*60}\n")
            
            # Save model and metrics
            if episode % save_interval == 0 and episode > 0:
                self.agent.save_model(self.model_save_path)
                metrics_path = f"metrics/training_metrics_ep{episode}.npy"
                self.metrics.save_metrics(metrics_path)
                print(f"💾 Model and metrics saved at episode {episode}")
        
        # Final results
        elapsed_time = time.time() - start_time
        stats = self.metrics.get_current_stats()
        
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED!")
        print("=" * 60)
        
        # Comprehensive final summary in requested format
        print("📊 FINAL TRAINING SUMMARY:")
        print("-" * 50)
        print(f"  🏆 Avg Reward: {stats['avg_reward']:8.2f}")
        print(f"  📏 Avg Distance: {stats['avg_distance']:6.4f}m")
        print(f"  ✅ Success Rate: {stats['success_rate']*100:5.1f}%")
        print(f"  🔥 Consecutive: {stats['consecutive_successes']:3d}")
        print(f"  🎯 Best Distance: {stats['best_distance']:6.4f}m")
        
        if self.agent_type == 'ddpg':
            print(f"  � Actor Loss: {stats['actor_loss']:8.4f}")
            print(f"  🧠 Critic Loss: {stats['critic_loss']:8.4f}")
        else:
            print(f"  🔍 Exploration: {stats['exploration_rate']:6.3f}")
        
        print(f"  ⏱️ Time: {elapsed_time/60:.1f}min")
        print("-" * 50)
        
        # Additional comprehensive metrics
        print("\n📈 DETAILED PERFORMANCE METRICS:")
        print("-" * 50)
        print(f"  📊 Total Episodes: {episodes}")
        print(f"  🎯 Best Episode Reward: {stats['best_episode_reward']:8.2f}")
        print(f"  � Total Successes: {stats['total_successes']}")
        print(f"  📈 Avg Improvement: {stats['avg_improvement']:+6.4f}m/ep")
        print(f"  ⚡ Efficiency: {stats['efficiency']:6.2f} reward/step")
        print(f"  🔄 Episodes/min: {episodes/(elapsed_time/60):.1f}")
        
        # Performance evaluation using benchmarks
        from advanced_config import evaluate_performance
        evaluation = evaluate_performance(stats['avg_reward'], stats['avg_distance'], stats['success_rate'])
        
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        print("-" * 50)
        print(f"  🏆 Reward Level: {evaluation['reward_level'].upper()}")
        print(f"  📏 Distance Level: {evaluation['distance_level'].upper()}")
        print(f"  ✅ Success Level: {evaluation['success_level'].upper()}")
        print(f"  🌟 Overall Assessment: {evaluation['overall'].upper()}")
        
        # Training efficiency metrics
        total_steps = sum(self.metrics.episode_lengths)
        print(f"\n⚙️ TRAINING EFFICIENCY:")
        print("-" * 50)
        print(f"  🔢 Total Steps: {total_steps}")
        print(f"  📊 Avg Steps/Episode: {np.mean(self.metrics.episode_lengths):.1f}")
        print(f"  ⚡ Steps/second: {total_steps/elapsed_time:.1f}")
        print(f"  🧠 Experience/Buffer: {len(self.agent.memory) if hasattr(self.agent, 'memory') else 'N/A'}")
        
        # Learning progress summary
        if len(self.metrics.episode_rewards) > 10:
            early_reward = np.mean(self.metrics.episode_rewards[:5])
            late_reward = np.mean(self.metrics.episode_rewards[-5:])
            reward_improvement = late_reward - early_reward
            
            early_distance = np.mean(self.metrics.distances_to_target[:5])
            late_distance = np.mean(self.metrics.distances_to_target[-5:])
            distance_improvement = early_distance - late_distance
            
            print(f"\n📈 LEARNING PROGRESS:")
            print("-" * 50)
            print(f"  🏆 Reward Improvement: {reward_improvement:+8.2f}")
            print(f"  📏 Distance Improvement: {distance_improvement:+6.4f}m")
            print(f"  📊 Learning Rate: {reward_improvement/episodes:.2f} reward/episode")
        
        # Intelligent recommendations based on training results
        try:
            from advanced_config import get_training_recommendations
            recommendations = get_training_recommendations(
                stats['avg_reward'], stats['avg_distance'], stats['success_rate'],
                stats['actor_loss'], stats['critic_loss'], episodes
            )
            
            print(f"\n💡 INTELLIGENT RECOMMENDATIONS:")
            print("-" * 50)
            for rec in recommendations:
                print(f"  {rec}")
                
        except ImportError:
            # Fallback to simple recommendations
            print(f"\n💡 BASIC RECOMMENDATIONS:")
            print("-" * 50)
            if stats['success_rate'] == 0:
                if stats['avg_distance'] > 0.30:
                    print("  📏 Focus on getting closer to targets (currently >30cm)")
                    print("  🔧 Consider increasing milestone rewards")
                    print("  📈 Train for more episodes (100-200)")
                elif stats['avg_distance'] > 0.15:
                    print("  🎯 Good progress! Distance improving, continue training")
                    print("  📈 Try 50-100 more episodes")
                else:
                    print("  🌟 Excellent distance! Success should come soon")
                    print("  📈 Continue training, you're very close!")
            else:
                print("  ✅ Great! Achieving some successes")
                print("  📈 Continue training to improve consistency")
            
            if evaluation['overall'] in ['poor', 'bad']:
                print("  ⚠️ Consider adjusting hyperparameters")
                print("  🔧 Try the problem-specific configurations in advanced_config.py")
            elif evaluation['overall'] == 'excellent':
                print("  🎉 Excellent performance! Consider more challenging targets")
                print("  🚀 Ready for hardware deployment")
        
        # Generate final plots
        final_plot_path = f"plots/final_training_results.png"
        self.metrics.plot_training_progress(save_path=final_plot_path)
        
        # Save final metrics
        final_metrics_path = "metrics/final_training_metrics.npy"
        self.metrics.save_metrics(final_metrics_path)
        
        print("\n" + "=" * 60)
        print("📁 FILES SAVED:")
        print(f"   📊 Plots: {final_plot_path}")
        print(f"   📈 Metrics: {final_metrics_path}")
        print(f"   🤖 Model: {self.model_save_path}")
        print("=" * 60)
        
    def _discretize_action(self, action_index: int) -> np.ndarray:
        """Convert discrete action index to continuous action"""
        action_discretization = 5
        actions = []
        remaining = action_index
        
        for _ in range(self.num_joints):
            actions.append(remaining % action_discretization)
            remaining //= action_discretization
        
        continuous_actions = []
        for discrete_val in actions:
            continuous_val = -1 + 2 * discrete_val / (action_discretization - 1)
            continuous_actions.append(continuous_val)
        
        return np.array(continuous_actions)

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Enhanced Robot Arm RL Training')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'demo'],
                       help='Mode: train, test, or demo')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes')
    parser.add_argument('--agent', type=str, default='ddpg',
                       choices=['ddpg', 'dqn'],
                       help='RL agent type')
    parser.add_argument('--no-robot', action='store_true',
                       help='Run without physical robot (simulation only)')
    parser.add_argument('--render-interval', type=int, default=10,
                       help='Print progress every N episodes')
    parser.add_argument('--fast', action='store_true',
                       help='Enable fast mode - reduced output for maximum speed')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = EnhancedRobotArmTrainer(
        use_physical_robot=not args.no_robot,
        agent_type=args.agent
    )
    
    if args.mode == 'train':
        trainer.train(
            episodes=args.episodes,
            render_interval=args.render_interval,
            fast_mode=args.fast
        )
    else:
        print("Test and demo modes coming soon!")

if __name__ == "__main__":
    main()
