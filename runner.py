import gym
import torch
import cv2
import numpy as np
from DuelingDDQN import DuelingDDQN  # Ensure this is the correct import for your model

def preprocess_observation(obs_tuple, new_size=(84, 84)):
    """
    Preprocess the observation (frame):
    1. Convert to grayscale if necessary
    2. Resize
    3. Normalize pixel values
    """
    if obs_tuple is None:
        return np.zeros((1, new_size[0], new_size[1]), dtype=np.float32)

    obs = obs_tuple[0]  # Extract the image from the tuple

    # Check if the image is already grayscale
    if obs.ndim == 3 and obs.shape[2] == 3:
        # Convert to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    else:
        gray = obs  # If already grayscale

    # Resize
    resized = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)

    # Normalize pixel values to be between 0 and 1
    normalized = resized / 255.0

    # Add channel dimension
    processed = np.expand_dims(normalized, axis=0)

    return processed

def load_checkpoint(filename="./checkpoint/pong_checkpoint_2.pth"):
    return torch.load(filename)

def main():
    # Load the environment
    env = gym.make('ALE/Pong-v5',render_mode="human")
    env.reset()

    # Load the model
    checkpoint = load_checkpoint("./checkpoint/pong_checkpoint_2.pth")
    policy_model = DuelingDDQN(84, 84, 6)  # Ensure these dimensions match your model
    policy_model.load_state_dict(checkpoint['policy_model_state_dict'])
    policy_model.eval()

    done = False
    while not done:
        # Render the environment
        env.render()

        # Preprocess the current state
        current_state = preprocess_observation(env.render())
        current_state = torch.tensor(current_state, dtype=torch.float32)

        # Get action from the model
        with torch.no_grad():
            # random 
            # action = env.action_space.sample()
            action = policy_model(current_state.unsqueeze(0)).max(1)[1].view(1, 1).item()

        # Take a step in the environment using the chosen action
        _, _, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        if done:
            break

    env.close()

if __name__ == "__main__":
    main()
