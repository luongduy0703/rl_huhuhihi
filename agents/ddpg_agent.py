import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from agents.base_agent import BaseAgent
import os

class DDPGAgent(BaseAgent):
    """
    Deep Deterministic Policy Gradient agent for continuous control (OOP module)
    """
    def __init__(self, state_size, action_size, actor_lr=0.00005, critic_lr=0.001, gamma=0.99,
                 tau=0.001, batch_size=64, memory_size=100000, noise_std=0.1):
        super().__init__(state_size, action_size, memory_size)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.target_actor = self._build_actor()
        self.target_critic = self._build_critic()
        self.actor_optimizer = keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = keras.optimizers.Adam(critic_lr)
        self.update_target_networks(tau=1.0)
        self.actor_loss_history = []
        self.critic_loss_history = []

    def _build_actor(self):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.action_size, activation='tanh')(x)
        model = keras.Model(inputs, outputs)
        return model

    def _build_critic(self):
        state_input = layers.Input(shape=(self.state_size,))
        action_input = layers.Input(shape=(self.action_size,))
        state_h1 = layers.Dense(256, activation='relu')(state_input)
        state_h1 = layers.BatchNormalization()(state_h1)
        state_h2 = layers.Dense(128)(state_h1)
        action_h1 = layers.Dense(128)(action_input)
        concat = layers.Concatenate()([state_h2, action_h1])
        concat_h1 = layers.Dense(128, activation='relu')(concat)
        concat_h1 = layers.BatchNormalization()(concat_h1)
        concat_h2 = layers.Dense(64, activation='relu')(concat_h1)
        outputs = layers.Dense(1)(concat_h2)
        model = keras.Model([state_input, action_input], outputs)
        return model

    def act(self, state, add_noise=True):
        state = state.reshape(1, -1)
        action = self.actor(state)[0]
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=self.action_size)
            action = np.clip(action + noise, -1, 1)
        if hasattr(action, 'numpy'):
            return action.numpy()
        else:
            return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None, None
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch]).reshape(-1, 1)
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch]).reshape(-1, 1)
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            target_q = self.target_critic([next_states, target_actions])
            y = rewards + self.gamma * target_q * (1 - dones)
            q_value = self.critic([states, actions])
            critic_loss = tf.reduce_mean(tf.square(y - q_value))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic([states, actions_pred]))
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.update_target_networks()
        self.actor_loss_history.append(float(actor_loss))
        self.critic_loss_history.append(float(critic_loss))
        return float(actor_loss), float(critic_loss)

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, param in zip(self.target_actor.variables, self.actor.variables):
            target_param.assign(tau * param + (1 - tau) * target_param)
        for target_param, param in zip(self.target_critic.variables, self.critic.variables):
            target_param.assign(tau * param + (1 - tau) * target_param)

    def save_model(self, filepath):
        self.actor.save_weights(f"{filepath}_actor_weights.h5")
        self.critic.save_weights(f"{filepath}_critic_weights.h5")
        params = {
            'actor_loss_history': self.actor_loss_history,
            'critic_loss_history': self.critic_loss_history,
            'reward_history': self.reward_history
        }
        self.save_params(f"{filepath}_params.pkl", params)
        print(f"DDPG model saved to {filepath}")

    def load_model(self, filepath):
        try:
            self.actor.load_weights(f"{filepath}_actor_weights.h5")
            self.critic.load_weights(f"{filepath}_critic_weights.h5")
            self.update_target_networks(tau=1.0)
            params = self.load_params(f"{filepath}_params.pkl")
            if params:
                self.actor_loss_history = params.get('actor_loss_history', [])
                self.critic_loss_history = params.get('critic_loss_history', [])
                self.reward_history = params.get('reward_history', [])
            print(f"DDPG model loaded from {filepath}")
            return True
        except:
            print(f"Failed to load DDPG model from {filepath}")
            return False
