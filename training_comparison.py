#!/usr/bin/env python3
"""
Training Comparison Tool - Compare Original vs Enhanced System
Demonstrates the improvements achieved
"""

import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import argparse

class TrainingComparator:
    """Compare training performance between different configurations"""
    
    def __init__(self):
        self.results = {}
        
    def run_training_session(self, script_name: str, episodes: int = 25, 
                           label: str = None) -> Dict:
        """Run a training session and capture results"""
        
        if label is None:
            label = script_name
            
        print(f"\n🚀 Starting {label} training session...")
        print(f"📊 Episodes: {episodes}")
        print("-" * 50)
        
        # Build command
        cmd = [
            'python3', script_name,
            '--mode', 'train',
            '--no-robot',
            '--episodes', str(episodes),
            '--render-interval', '5'
        ]
        
        # Run training and capture output
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            end_time = time.time()
            
            if result.returncode == 0:
                # Parse results from output
                output_lines = result.stdout.split('\n')
                metrics = self._parse_training_output(output_lines)
                metrics['duration'] = end_time - start_time
                metrics['success'] = True
                metrics['script'] = script_name
                metrics['label'] = label
                
                print(f"✅ {label} completed successfully!")
                print(f"⏱️ Duration: {metrics['duration']:.1f}s")
                print(f"🏆 Final Reward: {metrics.get('final_reward', 'N/A')}")
                print(f"📏 Final Distance: {metrics.get('final_distance', 'N/A')}")
                
            else:
                print(f"❌ {label} failed!")
                print(f"Error: {result.stderr}")
                metrics = {'success': False, 'error': result.stderr}
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {label} timed out after 10 minutes")
            metrics = {'success': False, 'error': 'Timeout'}
            
        self.results[label] = metrics
        return metrics
    
    def _parse_training_output(self, output_lines: List[str]) -> Dict:
        """Parse training metrics from output"""
        metrics = {
            'rewards': [],
            'distances': [],
            'actor_losses': [],
            'critic_losses': [],
            'final_reward': None,
            'final_distance': None,
            'final_success_rate': None
        }
        
        for line in output_lines:
            line = line.strip()
            
            # Parse episode metrics
            if 'Avg Reward:' in line:
                try:
                    reward_str = line.split('Avg Reward:')[1].split()[0]
                    reward = float(reward_str)
                    metrics['rewards'].append(reward)
                except:
                    pass
                    
            if 'Avg Distance:' in line:
                try:
                    distance_str = line.split('Avg Distance:')[1].split('m')[0].strip()
                    distance = float(distance_str)
                    metrics['distances'].append(distance)
                except:
                    pass
                    
            if 'Actor Loss:' in line:
                try:
                    loss_str = line.split('Actor Loss:')[1].split()[0]
                    loss = float(loss_str)
                    metrics['actor_losses'].append(loss)
                except:
                    pass
                    
            if 'Critic Loss:' in line:
                try:
                    loss_str = line.split('Critic Loss:')[1].split()[0]
                    loss = float(loss_str)
                    metrics['critic_losses'].append(loss)
                except:
                    pass
            
            # Parse final results
            if 'Final Average Reward:' in line:
                try:
                    metrics['final_reward'] = float(line.split('Final Average Reward:')[1].strip())
                except:
                    pass
                    
            if 'Best Distance Achieved:' in line:
                try:
                    distance_str = line.split('Best Distance Achieved:')[1].split('m')[0].strip()
                    metrics['final_distance'] = float(distance_str)
                except:
                    pass
                    
            if 'Final Success Rate:' in line:
                try:
                    rate_str = line.split('Final Success Rate:')[1].split('%')[0].strip()
                    metrics['final_success_rate'] = float(rate_str)
                except:
                    pass
        
        return metrics
    
    def compare_systems(self, episodes: int = 25):
        """Compare original vs enhanced system"""
        
        print("🔬 ROBOT ARM RL SYSTEM COMPARISON")
        print("=" * 60)
        print(f"📊 Training Episodes: {episodes}")
        print(f"🖥️ Mode: Simulation (no hardware)")
        print(f"⏱️ Estimated Time: ~{episodes * 0.3:.1f} minutes per system")
        print("=" * 60)
        
        # Run original system
        original_results = self.run_training_session(
            'main.py', episodes, 'Original System'
        )
        
        time.sleep(2)  # Brief pause between runs
        
        # Run enhanced system
        enhanced_results = self.run_training_session(
            'enhanced_trainer.py', episodes, 'Enhanced System'
        )
        
        # Generate comparison
        self._generate_comparison_report()
        self._plot_comparison()
        
    def _generate_comparison_report(self):
        """Generate detailed comparison report"""
        
        print("\n" + "=" * 60)
        print("📊 TRAINING COMPARISON RESULTS")
        print("=" * 60)
        
        if 'Original System' not in self.results or 'Enhanced System' not in self.results:
            print("❌ Cannot generate comparison - missing results")
            return
            
        orig = self.results['Original System']
        enh = self.results['Enhanced System']
        
        if not orig.get('success') or not enh.get('success'):
            print("❌ Cannot compare - one or both training sessions failed")
            return
        
        print("\n📈 PERFORMANCE METRICS:")
        print("-" * 40)
        
        # Reward comparison
        orig_reward = orig.get('final_reward', 0)
        enh_reward = enh.get('final_reward', 0)
        reward_improvement = ((enh_reward - orig_reward) / abs(orig_reward) * 100) if orig_reward != 0 else 0
        
        print(f"🏆 Average Reward:")
        print(f"   Original: {orig_reward:8.2f}")
        print(f"   Enhanced: {enh_reward:8.2f}")
        print(f"   Improvement: {reward_improvement:+6.1f}%")
        
        # Distance comparison
        orig_distance = orig.get('final_distance', 0)
        enh_distance = enh.get('final_distance', 0)
        distance_improvement = ((orig_distance - enh_distance) / orig_distance * 100) if orig_distance != 0 else 0
        
        print(f"\n📏 Distance to Target:")
        print(f"   Original: {orig_distance:6.4f}m")
        print(f"   Enhanced: {enh_distance:6.4f}m")
        print(f"   Improvement: {distance_improvement:+6.1f}%")
        
        # Success rate comparison
        orig_success = orig.get('final_success_rate', 0)
        enh_success = enh.get('final_success_rate', 0)
        
        print(f"\n✅ Success Rate:")
        print(f"   Original: {orig_success:5.1f}%")
        print(f"   Enhanced: {enh_success:5.1f}%")
        
        # Training time comparison
        orig_time = orig.get('duration', 0)
        enh_time = enh.get('duration', 0)
        
        print(f"\n⏱️ Training Time:")
        print(f"   Original: {orig_time:6.1f}s")
        print(f"   Enhanced: {enh_time:6.1f}s")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print("-" * 40)
        
        improvements = []
        if reward_improvement > 50:
            improvements.append("🏆 Major reward improvement")
        if distance_improvement > 20:
            improvements.append("📏 Significant distance improvement")
        if enh_success > orig_success:
            improvements.append("✅ Better success rate")
            
        if improvements:
            print("✅ Enhanced system shows clear improvements:")
            for improvement in improvements:
                print(f"   {improvement}")
        else:
            print("⚠️ Results inconclusive - may need longer training")
            
    def _plot_comparison(self):
        """Generate comparison plots"""
        
        if 'Original System' not in self.results or 'Enhanced System' not in self.results:
            return
            
        orig = self.results['Original System']
        enh = self.results['Enhanced System']
        
        if not orig.get('success') or not enh.get('success'):
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training System Comparison: Original vs Enhanced', fontsize=16, fontweight='bold')
        
        # Rewards comparison
        if orig.get('rewards') and enh.get('rewards'):
            axes[0, 0].plot(orig['rewards'], 'r-', alpha=0.7, label='Original System', linewidth=2)
            axes[0, 0].plot(enh['rewards'], 'g-', alpha=0.7, label='Enhanced System', linewidth=2)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Training Progress')
            axes[0, 0].set_ylabel('Average Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Distance comparison
        if orig.get('distances') and enh.get('distances'):
            axes[0, 1].plot(orig['distances'], 'r-', alpha=0.7, label='Original System', linewidth=2)
            axes[0, 1].plot(enh['distances'], 'g-', alpha=0.7, label='Enhanced System', linewidth=2)
            axes[0, 1].axhline(y=0.05, color='orange', linestyle='--', label='Success Threshold')
            axes[0, 1].set_title('Distance to Target')
            axes[0, 1].set_xlabel('Training Progress')
            axes[0, 1].set_ylabel('Distance (m)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Actor losses comparison
        if orig.get('actor_losses') and enh.get('actor_losses'):
            axes[1, 0].plot(orig['actor_losses'], 'r-', alpha=0.7, label='Original System', linewidth=2)
            axes[1, 0].plot(enh['actor_losses'], 'g-', alpha=0.7, label='Enhanced System', linewidth=2)
            axes[1, 0].set_title('Actor Loss')
            axes[1, 0].set_xlabel('Training Progress')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Critic losses comparison
        if orig.get('critic_losses') and enh.get('critic_losses'):
            axes[1, 1].plot(orig['critic_losses'], 'r-', alpha=0.7, label='Original System', linewidth=2)
            axes[1, 1].plot(enh['critic_losses'], 'g-', alpha=0.7, label='Enhanced System', linewidth=2)
            axes[1, 1].set_title('Critic Loss')
            axes[1, 1].set_xlabel('Training Progress')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/system_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Comparison plots saved to: plots/system_comparison.png")
        plt.show()

def main():
    """Main comparison function"""
    parser = argparse.ArgumentParser(description='Compare Robot Arm RL Training Systems')
    parser.add_argument('--episodes', type=int, default=25,
                       help='Number of training episodes for each system')
    parser.add_argument('--quick', action='store_true',
                       help='Quick comparison with fewer episodes')
    
    args = parser.parse_args()
    
    episodes = 10 if args.quick else args.episodes
    
    comparator = TrainingComparator()
    comparator.compare_systems(episodes)

if __name__ == "__main__":
    main()
