import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque
import pickle
import os
from typing import List, Tuple, Optional

class DQNAgent:
    """
    Deep Q-Network agent for robot arm control
    """
    
    def __init__(self, 
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 32,
                 memory_size: int = 10000):
        """
        Initialize DQN Agent
        
        Args:
            state_size: Size of state space
            action_size: Size of action space  
            learning_rate: Learning rate for neural network
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            batch_size: Batch size for training
            memory_size: Size of experience replay buffer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Neural networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        
        # Update target network
        self.update_target_network()
        
        # Training metrics
        self.loss_history = []
        self.reward_history = []
        
    def _build_model(self) -> keras.Model:
        """Build neural network model"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_dim=self.state_size),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self) -> Optional[float]:
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        # Current Q values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Next Q values from target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Update Q values
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        history = self.q_network.fit(states, current_q_values, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.loss_history.append(loss)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def update_target_network(self):
        """Update target network with main network weights"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def save_model(self, filepath: str):
        """Save model weights and parameters"""
        self.q_network.save_weights(f"{filepath}_weights.h5")
        
        # Save agent parameters
        params = {
            'epsilon': self.epsilon,
            'loss_history': self.loss_history,
            'reward_history': self.reward_history
        }
        
        with open(f"{filepath}_params.pkl", 'wb') as f:
            pickle.dump(params, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights and parameters"""
        if os.path.exists(f"{filepath}_weights.h5"):
            self.q_network.load_weights(f"{filepath}_weights.h5")
            self.update_target_network()
            
            # Load agent parameters
            if os.path.exists(f"{filepath}_params.pkl"):
                with open(f"{filepath}_params.pkl", 'rb') as f:
                    params = pickle.load(f)
                    self.epsilon = params.get('epsilon', self.epsilon)
                    self.loss_history = params.get('loss_history', [])
                    self.reward_history = params.get('reward_history', [])
            
            print(f"Model loaded from {filepath}")
            return True
        else:
            print(f"No model found at {filepath}")
            return False

class DDPGAgent:
    """
    Deep Deterministic Policy Gradient agent for continuous control
    Better suited for robot arm control with continuous actions
    """
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 actor_lr: float = 0.0001,  # Reduced to prevent increasing loss
                 critic_lr: float = 0.001,   # Reduced from 0.002
                 gamma: float = 0.99,
                 tau: float = 0.001,         # Softer target updates 
                 batch_size: int = 64,
                 memory_size: int = 100000,
                 noise_std: float = 0.1):    # Reduced exploration noise
        """
        Initialize DDPG Agent
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            actor_lr: Learning rate for actor network
            critic_lr: Learning rate for critic network
            gamma: Discount factor
            tau: Soft update parameter
            batch_size: Batch size for training
            memory_size: Size of experience replay buffer
            noise_std: Standard deviation for exploration noise
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.target_actor = self._build_actor()
        self.target_critic = self._build_critic()
        
        # Optimizers
        self.actor_optimizer = keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = keras.optimizers.Adam(critic_lr)
        
        # Initialize target networks
        self.update_target_networks(tau=1.0)
        
        # Training metrics
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.reward_history = []
    
    def _build_actor(self) -> keras.Model:
        """Build actor network"""
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.action_size, activation='tanh')(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def _build_critic(self) -> keras.Model:
        """Build critic network"""
        state_input = layers.Input(shape=(self.state_size,))
        action_input = layers.Input(shape=(self.action_size,))
        
        # State pathway
        state_h1 = layers.Dense(256, activation='relu')(state_input)
        state_h1 = layers.BatchNormalization()(state_h1)
        state_h2 = layers.Dense(128)(state_h1)
        
        # Action pathway
        action_h1 = layers.Dense(128)(action_input)
        
        # Combine state and action
        concat = layers.Concatenate()([state_h2, action_h1])
        concat_h1 = layers.Dense(128, activation='relu')(concat)
        concat_h1 = layers.BatchNormalization()(concat_h1)
        concat_h2 = layers.Dense(64, activation='relu')(concat_h1)
        outputs = layers.Dense(1)(concat_h2)
        
        model = keras.Model([state_input, action_input], outputs)
        return model
    
    def act(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action using actor network with exploration noise"""
        state = state.reshape(1, -1)
        action = self.actor(state)[0]
        
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=self.action_size)
            action = np.clip(action + noise, -1, 1)
        
        # Convert tensor to numpy if needed
        if hasattr(action, 'numpy'):
            return action.numpy()
        else:
            return action
    
    def remember(self, state: np.ndarray, action: np.ndarray, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self) -> Tuple[Optional[float], Optional[float]]:
        """Train both actor and critic networks"""
        if len(self.memory) < self.batch_size:
            return None, None
        
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch]).reshape(-1, 1)
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch]).reshape(-1, 1)
        
        # Train critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            target_q = self.target_critic([next_states, target_actions])
            y = rewards + self.gamma * target_q * (1 - dones)
            q_value = self.critic([states, actions])
            critic_loss = tf.reduce_mean(tf.square(y - q_value))
        
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # Train actor
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic([states, actions_pred]))
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Update target networks
        self.update_target_networks()
        
        # Store losses
        self.actor_loss_history.append(float(actor_loss))
        self.critic_loss_history.append(float(critic_loss))
        
        return float(actor_loss), float(critic_loss)
    
    def update_target_networks(self, tau: Optional[float] = None):
        """Soft update target networks"""
        if tau is None:
            tau = self.tau
        
        # Update target actor
        for target_param, param in zip(self.target_actor.variables, self.actor.variables):
            target_param.assign(tau * param + (1 - tau) * target_param)
        
        # Update target critic
        for target_param, param in zip(self.target_critic.variables, self.critic.variables):
            target_param.assign(tau * param + (1 - tau) * target_param)
    
    def save_model(self, filepath: str):
        """Save model weights and parameters"""
        self.actor.save_weights(f"{filepath}_actor_weights.h5")
        self.critic.save_weights(f"{filepath}_critic_weights.h5")
        
        params = {
            'actor_loss_history': self.actor_loss_history,
            'critic_loss_history': self.critic_loss_history,
            'reward_history': self.reward_history
        }
        
        with open(f"{filepath}_params.pkl", 'wb') as f:
            pickle.dump(params, f)
        
        print(f"DDPG model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights and parameters"""
        try:
            self.actor.load_weights(f"{filepath}_actor_weights.h5")
            self.critic.load_weights(f"{filepath}_critic_weights.h5")
            self.update_target_networks(tau=1.0)
            
            if os.path.exists(f"{filepath}_params.pkl"):
                with open(f"{filepath}_params.pkl", 'rb') as f:
                    params = pickle.load(f)
                    self.actor_loss_history = params.get('actor_loss_history', [])
                    self.critic_loss_history = params.get('critic_loss_history', [])
                    self.reward_history = params.get('reward_history', [])
            
            print(f"DDPG model loaded from {filepath}")
            return True
        except:
            print(f"Failed to load DDPG model from {filepath}")
            return False
