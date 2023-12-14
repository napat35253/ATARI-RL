# Train.py
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(policy_model, target_model, optimizer, replay_buffer, batch_size, gamma):
    policy_model = policy_model.to(device)
    target_model = target_model.to(device)

    if len(replay_buffer) < batch_size:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.bool).to(device)

    current_q_values = policy_model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    # Double Q-Learning update
    next_actions = policy_model(next_states).max(1)[1]
    next_q_values = target_model(next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
    next_q_values[dones] = 0.0

    expected_q_values = rewards + gamma * next_q_values

    loss = F.mse_loss(current_q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

