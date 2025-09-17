import numpy as np
import random
from collections import deque
import pickle
import os
from typing import Optional

class BaseAgent:
    """
    Base class for RL agents (OOP module)
    """
    def __init__(self, state_size: int, action_size: int, memory_size: int = 10000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.reward_history = []

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save_params(self, filepath: str, params: dict):
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, filepath: str) -> Optional[dict]:
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None

    def save_replay_buffer(self, filepath: str):
        """Lưu replay buffer ra file."""
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.memory), f)

    def load_replay_buffer(self, filepath: str):
        """Tải lại replay buffer từ file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                buffer = pickle.load(f)
                self.memory = deque(buffer, maxlen=self.memory.maxlen)
