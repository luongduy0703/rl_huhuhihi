#!/usr/bin/env python3
"""
Training Analysis and Improvement Recommendations
Based on your specific training observations
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

class TrainingAnalyzer:
    """Analyze training patterns and provide improvement recommendations"""
    
    def __init__(self):
        self.analysis_results = {}
        
    def analyze_reward_pattern(self, rewards: List[float]) -> Dict:
        """Analyze reward patterns (-30 to -15 fluctuating)"""
        
        analysis = {
            'pattern_type': 'unstable_learning',
            'issues': [],
            'recommendations': []
        }
        
        # Check for high negative rewards
        avg_reward = np.mean(rewards)
        if avg_reward < -20:
            analysis['issues'].append("Rewards consistently very negative")
            analysis['recommendations'].extend([
                "Reduce movement penalties in reward function",
                "Increase target reaching bonus",
                "Add intermediate rewards for getting closer to target",
                "Check if target positions are actually reachable"
            ])
        
        # Check for high fluctuation
        reward_std = np.std(rewards)
        if reward_std > 10:
            analysis['issues'].append("High reward fluctuation indicates unstable learning")
            analysis['recommendations'].extend([
                "Reduce exploration noise (decrease epsilon or noise_std)",
                "Increase replay buffer size for more stable learning",
                "Use reward normalization or clipping"
            ])
        
        return analysis
    
    def analyze_distance_pattern(self, distances: List[float]) -> Dict:
        """Analyze distance to target (0.32-0.40m consistently)"""
        
        analysis = {
            'pattern_type': 'no_improvement',
            'issues': [],
            'recommendations': []
        }
        
        avg_distance = np.mean(distances)
        distance_improvement = distances[-10:] - distances[:10] if len(distances) > 20 else [0]
        
        if avg_distance > 0.15:  # 15cm is still quite far
            analysis['issues'].append("Distance to target too large (30-40cm)")
            analysis['recommendations'].extend([
                "Check forward kinematics accuracy",
                "Verify target positions are within robot workspace",
                "Increase reward gradient for getting closer",
                "Add curriculum learning (start with easier, closer targets)"
            ])
        
        if np.mean(distance_improvement) > -0.01:  # No significant improvement
            analysis['issues'].append("No distance improvement over time")
            analysis['recommendations'].extend([
                "Implement shaped rewards based on distance improvement",
                "Use hindsight experience replay (HER)",
                "Check if action space is appropriate for required precision"
            ])
        
        return analysis
    
    def analyze_loss_patterns(self, actor_losses: List[float], critic_losses: List[float]) -> Dict:
        """Analyze loss patterns"""
        
        analysis = {
            'actor_analysis': {},
            'critic_analysis': {},
            'recommendations': []
        }
        
        # Actor loss analysis (increasing linear function)
        if len(actor_losses) > 10:
            actor_trend = np.polyfit(range(len(actor_losses)), actor_losses, 1)[0]
            if actor_trend > 0:
                analysis['actor_analysis'] = {
                    'pattern': 'increasing',
                    'issue': 'Policy may be becoming unstable'
                }
                analysis['recommendations'].extend([
                    "Reduce actor learning rate (try 0.0001 instead of 0.001)",
                    "Implement gradient clipping",
                    "Add L2 regularization to actor network",
                    "Use softer target network updates (smaller tau)"
                ])
        
        # Critic loss analysis (small, jumps around)
        if len(critic_losses) > 10:
            critic_std = np.std(critic_losses)
            critic_mean = np.mean(critic_losses)
            
            analysis['critic_analysis'] = {
                'mean': critic_mean,
                'stability': 'stable' if critic_std < 0.01 else 'unstable'
            }
            
            if critic_std > 0.01:
                analysis['recommendations'].extend([
                    "Check for numerical instability in critic network",
                    "Use batch normalization in critic",
                    "Implement experience replay prioritization"
                ])
        
        return analysis
    
    def generate_improvement_config(self) -> Dict:
        """Generate improved configuration based on analysis"""
        
        improved_config = {
            # Reward function improvements
            'reward_config': {
                'distance_weight': -2.0,      # Stronger penalty for being far
                'improvement_bonus': 5.0,     # Bonus for getting closer
                'target_bonus': 20.0,         # Increased bonus for reaching target
                'movement_penalty': -0.005,   # Reduced movement penalty
                'step_penalty': -0.01,        # Small step penalty to encourage efficiency
            },
            
            # Learning parameters
            'learning_config': {
                'actor_lr': 0.0001,           # Reduced from 0.001
                'critic_lr': 0.001,           # Reduced from 0.002
                'batch_size': 64,             # Increased for stability
                'memory_size': 50000,         # Larger replay buffer
                'tau': 0.001,                 # Softer target updates
                'gamma': 0.99,                # Discount factor
            },
            
            # Exploration parameters
            'exploration_config': {
                'noise_std': 0.05,            # Reduced exploration noise
                'noise_decay': 0.995,         # Decay noise over time
                'min_noise': 0.01,            # Minimum noise level
            },
            
            # Environment improvements
            'environment_config': {
                'max_episode_steps': 150,     # Reduced to encourage efficiency
                'success_threshold': 0.05,    # 5cm success threshold
                'workspace_limits': 0.4,      # Limit workspace to reachable area
                'curriculum_learning': True,  # Start with easier targets
            }
        }
        
        return improved_config
    
    def create_diagnostic_plots(self, rewards: List[float], distances: List[float], 
                              actor_losses: List[float], critic_losses: List[float]):
        """Create diagnostic plots for your specific case"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Diagnostics - Your Specific Case', fontsize=16)
        
        episodes = range(len(rewards))
        
        # 1. Reward pattern analysis
        axes[0, 0].plot(episodes, rewards, 'b-', alpha=0.7, label='Episode Rewards')
        axes[0, 0].axhline(y=-15, color='g', linestyle='--', label='Target (-15)')
        axes[0, 0].axhline(y=-30, color='r', linestyle='--', label='Poor Performance (-30)')
        axes[0, 0].set_title('Reward Pattern: Logarithmic → Linear Fluctuation')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add trend analysis
        if len(rewards) > 10:
            # Fit logarithmic trend for early episodes
            early_episodes = episodes[:min(30, len(episodes)//2)]
            if len(early_episodes) > 5:
                log_fit = np.polyfit(np.log(np.array(early_episodes) + 1), 
                                   rewards[:len(early_episodes)], 1)
                log_trend = log_fit[0] * np.log(np.array(early_episodes) + 1) + log_fit[1]
                axes[0, 0].plot(early_episodes, log_trend, 'g--', alpha=0.8, label='Logarithmic Trend')
        
        # 2. Distance consistency analysis
        axes[0, 1].plot(episodes, distances, 'r-', alpha=0.7, label='Distance to Target')
        axes[0, 1].axhline(y=0.05, color='g', linestyle='--', label='Success Threshold (5cm)')
        axes[0, 1].axhspan(0.32, 0.40, alpha=0.3, color='orange', label='Your Observed Range')
        axes[0, 1].set_title('Distance: Too Consistent (No Improvement)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Distance (m)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Actor loss trend
        if actor_losses:
            axes[1, 0].plot(range(len(actor_losses)), actor_losses, 'orange', linewidth=2)
            # Fit linear trend
            if len(actor_losses) > 5:
                trend = np.polyfit(range(len(actor_losses)), actor_losses, 1)
                trend_line = np.poly1d(trend)
                axes[1, 0].plot(range(len(actor_losses)), trend_line(range(len(actor_losses))), 
                               'r--', alpha=0.8, label=f'Trend (slope: {trend[0]:.6f})')
                axes[1, 0].legend()
            axes[1, 0].set_title('Actor Loss: Increasing Linear (Concerning)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Actor Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Critic loss jumps
        if critic_losses:
            axes[1, 1].plot(range(len(critic_losses)), critic_losses, 'red', linewidth=2)
            axes[1, 1].set_title('Critic Loss: Small but Jumpy')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Critic Loss')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Highlight jumps
            for i, loss in enumerate(critic_losses):
                if loss < 0.001:  # Your observed 0.0 values
                    axes[1, 1].scatter(i, loss, color='blue', s=50, alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        return fig

def main():
    """Demonstrate analysis with simulated data matching user's observations"""
    
    analyzer = TrainingAnalyzer()
    
    # Simulate data matching user's description
    print("🔍 ANALYZING YOUR TRAINING PATTERNS")
    print("=" * 50)
    
    # Simulated reward pattern: logarithmic early, then fluctuating
    episodes = 100
    rewards = []
    for i in range(episodes):
        if i < 30:  # Logarithmic improvement phase
            base_reward = -30 + 15 * np.log(i + 1) / np.log(30)
        else:  # Linear fluctuation phase
            base_reward = -22 + 7 * np.sin(i * 0.3) + np.random.normal(0, 2)
        rewards.append(base_reward)
    
    # Consistent distances (0.32-0.40)
    distances = [0.36 + 0.04 * np.sin(i * 0.1) + np.random.normal(0, 0.02) for i in range(episodes)]
    
    # Increasing actor losses
    actor_losses = [0.001 + 0.0002 * i + np.random.normal(0, 0.0001) for i in range(episodes)]
    
    # Jumpy critic losses (small values)
    critic_losses = []
    for i in range(episodes):
        if np.random.random() < 0.1:  # 10% chance of jump to 0
            critic_losses.append(0.0)
        else:
            critic_losses.append(0.02 + 0.05 * np.random.random())
    
    # Analyze patterns
    reward_analysis = analyzer.analyze_reward_pattern(rewards)
    distance_analysis = analyzer.analyze_distance_pattern(distances)
    loss_analysis = analyzer.analyze_loss_patterns(actor_losses, critic_losses)
    
    # Print analysis
    print("🏆 REWARD ANALYSIS:")
    for issue in reward_analysis['issues']:
        print(f"  ❌ {issue}")
    for rec in reward_analysis['recommendations']:
        print(f"  💡 {rec}")
    
    print("\n📏 DISTANCE ANALYSIS:")
    for issue in distance_analysis['issues']:
        print(f"  ❌ {issue}")
    for rec in distance_analysis['recommendations']:
        print(f"  💡 {rec}")
    
    print("\n🧠 LOSS ANALYSIS:")
    print(f"  🎭 Actor: {loss_analysis['actor_analysis']}")
    print(f"  🧠 Critic: {loss_analysis['critic_analysis']}")
    for rec in loss_analysis['recommendations']:
        print(f"  💡 {rec}")
    
    # Generate improved config
    print("\n⚙️ IMPROVED CONFIGURATION:")
    improved_config = analyzer.generate_improvement_config()
    for category, configs in improved_config.items():
        print(f"\n  {category}:")
        for key, value in configs.items():
            print(f"    {key}: {value}")
    
    # Create diagnostic plots
    print("\n📊 Generating diagnostic plots...")
    analyzer.create_diagnostic_plots(rewards, distances, actor_losses, critic_losses)

if __name__ == "__main__":
    main()
