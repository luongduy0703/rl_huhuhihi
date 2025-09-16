import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from agents.base_agent import BaseAgent

class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent for robot arm control (OOP module)
    """
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=32, memory_size=10000):
        super().__init__(state_size, action_size, memory_size)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        self.loss_history = []

    def _build_model(self):
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_dim=self.state_size),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        current_q_values = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        history = self.q_network.fit(states, current_q_values, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.loss_history.append(loss)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def save_model(self, filepath):
        self.q_network.save_weights(f"{filepath}_weights.h5")
        params = {
            'epsilon': self.epsilon,
            'loss_history': self.loss_history,
            'reward_history': self.reward_history
        }
        self.save_params(f"{filepath}_params.pkl", params)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        if os.path.exists(f"{filepath}_weights.h5"):
            self.q_network.load_weights(f"{filepath}_weights.h5")
            self.update_target_network()
            params = self.load_params(f"{filepath}_params.pkl")
            if params:
                self.epsilon = params.get('epsilon', self.epsilon)
                self.loss_history = params.get('loss_history', [])
                self.reward_history = params.get('reward_history', [])
            print(f"Model loaded from {filepath}")
            return True
        else:
            print(f"No model found at {filepath}")
            return False
