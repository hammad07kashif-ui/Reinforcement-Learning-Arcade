"""
╔══════════════════════════════════════════════════════════════╗
║   NEURAL ARCADE - FLAPPY CNN MODEL  |  VISION UPGRADE (M2)   ║
║                                                              ║
║  Same DeepMind DQN architecture as Snake CNN.                ║
║  Input : 4 stacked 84x84 grayscale frames                    ║
║  Output: Q-value per action  (2 actions: glide / flap)       ║
╚══════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

FRAME_H     = 84
FRAME_W     = 84
FRAME_STACK = 4
CONV_FLAT   = 64 * 7 * 7   # 3136


class ConvQNet(nn.Module):
    """
    Convolutional DQN for Flappy Bird AI.

    Identical topology to the Snake CNN - only n_actions differs (2 vs 3).
    This lets both models share the same training code and Trainer class.

    Input  : (batch, 4, 84, 84) - 4 stacked greyscale frames, values ∈ [0,1]
    Output : (batch, 2)         - Q(s, glide),  Q(s, flap)
    """

    def __init__(self, n_actions: int = 2):
        super().__init__()
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32,          64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64,          64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(CONV_FLAT, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.fc2.weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def save(self, file_name: str = 'flappy_cnn_model.pth'):
        folder = './model'
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, file_name)
        torch.save(self.state_dict(), path)
        print(f"[OK] CNN weights saved → {path}")

    def load(self, file_name: str = 'flappy_cnn_model.pth', device: str = 'cpu'):
        path = os.path.join('./model', file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"[X] No checkpoint found at: {path}")
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        self.eval()
        print(f"[OK] CNN weights loaded ← {path}")


class ConvQTrainer:
    """
    Shared Bellman trainer for both Snake and Flappy ConvQNets.
    Uses Huber loss and gradient clipping for stable pixel-based training.
    """

    def __init__(self, model: ConvQNet, lr: float = 1e-4, gamma: float = 0.99):
        self.model     = model
        self.gamma     = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        self.criterion = nn.SmoothL1Loss()

    def train_step(self, state, action, reward, next_state, done):
        state      = torch.tensor(np.array(state),      dtype=torch.float32)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32)
        action     = torch.tensor(np.array(action),     dtype=torch.long)
        reward     = torch.tensor(np.array(reward),     dtype=torch.float32)

        if state.dim() == 3:
            state      = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action     = action.unsqueeze(0)
            reward     = reward.unsqueeze(0)
            done       = (done,)

        pred   = self.model(state)
        target = pred.clone().detach()

        with torch.no_grad():
            next_q = self.model(next_state)

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(next_q[idx])
            act_idx = torch.argmax(action[idx]).item()
            target[idx][act_idx] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()


if __name__ == '__main__':
    print("\n─── Flappy Bird ConvQNet Architecture ───")
    model = ConvQNet(n_actions=2)
    dummy = torch.zeros(1, FRAME_STACK, FRAME_H, FRAME_W)
    out   = model(dummy)
    print(f"Input  : {tuple(dummy.shape)}")
    print(f"Output : {tuple(out.shape)}  (Q-values for 2 actions)")
    total = sum(p.numel() for p in model.parameters())
    print(f"Params : {total:,}")
    print()
    print(model)
