# Train.py
import torch
import torch.nn.functional as F

def train(model, optimizer, replay_buffer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.bool)

    q_values = model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    next_q_values = model(next_states).max(1)[0]
    next_q_values[dones] = 0.0

    expected_q_values = rewards + gamma * next_q_values

    loss = F.mse_loss(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
