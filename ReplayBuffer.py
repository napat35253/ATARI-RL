# ReplayBuffer.py
from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        states = np.array([np.array(item[0], dtype=np.float32) for item in batch])
        actions = np.array([item[1] for item in batch])
        rewards = np.array([item[2] for item in batch])
        next_states = np.array([np.array(item[3], dtype=np.float32) for item in batch])
        dones = np.array([item[4] for item in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
