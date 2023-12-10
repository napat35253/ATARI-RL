# main.py
import gym
import torch.optim as optim
import torch
from DuelCNN import DuelCNN
from ReplayBuffer import ReplayBuffer
from Train import train
import cv2
import numpy as np
# import logging
import os
from tqdm import tqdm
from Logger import Logger

input_shape = (1, 84, 64)
n_actions = 6 
batch_size = 32
gamma = 0.99
epsilon = 0.001
n_episode = 100000

def preprocess_observation(obs_tuple, new_size=(84, 64), crop_top=20):
    """
    Preprocess the observation (frame):
    1. Crop the top part of the image
    2. Convert to grayscale if necessary
    3. Resize
    4. Normalize pixel values
    """
    if obs_tuple is None:
        return np.zeros((1, new_size[0], new_size[1]), dtype=np.float32)

    obs = obs_tuple[0]  # Extract the image from the tuple

    # Crop the top part of the image
    cropped_obs = obs[crop_top:, :, :] if obs.ndim == 3 else obs[crop_top:, :]

    # Check if the image is already grayscale
    if cropped_obs.ndim == 3 and cropped_obs.shape[2] == 3:
        # Convert to grayscale
        gray = cv2.cvtColor(cropped_obs, cv2.COLOR_RGB2GRAY)
    else:
        gray = cropped_obs  # If already grayscale

    # Resize
    resized = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)

    # Normalize pixel values to be between 0 and 1
    normalized = resized / 255.0

    # Add channel dimension
    processed = np.expand_dims(normalized, axis=0)

    return processed

def save_checkpoint(state, filename="./checkpoint/pong_checkpoint.pth"):
    torch.save(state, filename)

def load_checkpoint(filename="./checkpoint/pong_checkpoint.pth"):
    return torch.load(filename)

# Initialize environment, model, optimizer, and replay buffer
# env = gym.make('ALE/Pong-v5', render_mode="human")
env = gym.make('ALE/Pong-v5', render_mode="rgb_array")
model = DuelCNN(input_shape[1],input_shape[2], n_actions)
optimizer = optim.Adam(model.parameters(), lr=epsilon)
replay_buffer = ReplayBuffer(10000)

total_steps = 0

# Setup logging
logger = Logger('./log/training_log.log')


checkpoint_file = "./checkpoint/pong_checkpoint_0.pth"

if os.path.isfile(checkpoint_file):
    checkpoint = load_checkpoint(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_episode = checkpoint['episode']
    logger.info(f"Loaded checkpoint from episode {start_episode}")
else:
    start_episode = 0

# Initialize tqdm progress bar
progress_bar = tqdm(range(start_episode, n_episode), desc="Training Episodes")

for episode in progress_bar:

    state = preprocess_observation(env.reset())
    state = torch.tensor(state, dtype=torch.float32)
    total_reward = 0

    while True:
        if replay_buffer.__len__() > batch_size:
            action = model(state.unsqueeze(0)).max(1)[1].item()
        else:
            action = env.action_space.sample()

        raw_next_state, reward, terminated, truncated, info = env.step(action)
        next_state = preprocess_observation(raw_next_state)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        # Consider the episode done whether it's terminated or truncated
        done = terminated or truncated

        # Add to replay buffer only if next state is valid
        if next_state is not None:
            replay_buffer.push(state.numpy(), action, reward, next_state.numpy(), done)

        state = next_state
        total_reward += reward
        total_steps += 1

        if done:
            break

        train(model, optimizer, replay_buffer, batch_size, gamma)

    # Update tqdm description with the latest total reward
    progress_bar.set_description(f"Ep: {episode} Reward: {total_reward}")

    if episode % 20 == 0:
        save_checkpoint({
            'episode': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_file)

    logger.info(f'Episode: {episode}, Total Steps: {total_steps}, Total Reward: {total_reward}')
    # print(f'Episode: {episode}, Total Steps: {total_steps}, Total Reward: {total_reward}')

env.close()