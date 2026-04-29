"""
╔══════════════════════════════════════════════════════════════╗
║     NEURAL ARCADE - CNN MODEL  |  VISION UPGRADE (M2)        ║
║                                                              ║
║  Architecture: DeepMind-style DQN with 3 Conv layers.        ║
║  Input : 4 stacked 84x84 grayscale frames                    ║
║  Output: Q-value per action                                  ║
║                                                              ║
║  Replaces: Linear_QNet (11 scalars) → ConvQNet (4x84x84 px) ║
╚══════════════════════════════════════════════════════════════╝

Architecture Reference:
  Mnih et al. (2015) - "Human-level control through deep RL"
  DeepMind Atari DQN - nature14236

  Layer │ Type          │ Filters │ Kernel │ Stride │ Output
  ──────┼───────────────┼─────────┼────────┼────────┼────────────
    1   │ Conv2d        │ 32      │ 8x8    │ 4      │ 32x20x20
    2   │ Conv2d        │ 64      │ 4x4    │ 2      │ 64x9x9
    3   │ Conv2d        │ 64      │ 3x3    │ 1      │ 64x7x7
    -   │ Flatten       │ -       │ -      │ -      │ 3136
    4   │ Linear        │ 512     │ -      │ -      │ 512
    5   │ Linear        │ actions │ -      │ -      │ 3 (snake)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

# ── Frame dimensions (standard DQN - Atari paper) ─────────────
FRAME_H      = 84
FRAME_W      = 84
FRAME_STACK  = 4      # Number of consecutive frames stacked as "temporal" input
CONV_FLAT    = 64 * 7 * 7   # = 3136  (output of 3rd conv layer flattened)


class ConvQNet(nn.Module):
    """
    Convolutional Deep-Q Network for Snake AI.
    
    Input:  Tensor of shape (batch, 4, 84, 84) - 4 stacked grayscale frames
    Output: Tensor of shape (batch, n_actions) - Q-value for each action

    The three convolutional layers extract spatial features from raw pixels.
    The two fully-connected layers translate features into action Q-values.
    """

    def __init__(self, n_actions: int = 3):
        super().__init__()
        self.n_actions = n_actions

        # ── Convolutional Feature Extractor ──────────────────────
        self.conv1 = nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32,          64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64,          64, kernel_size=3, stride=1)

        # ── Fully-Connected Decision Layers ──────────────────────
        self.fc1 = nn.Linear(CONV_FLAT, 512)
        self.fc2 = nn.Linear(512, n_actions)

        # ── Weight initialisation (orthogonal - more stable than default) ─
        self._init_weights()

    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.fc2.weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: (batch, 4, 84, 84) float tensor, values normalised to [0, 1].
        """
        x = F.relu(self.conv1(x))   # → (batch, 32, 20, 20)
        x = F.relu(self.conv2(x))   # → (batch, 64,  9,  9)
        x = F.relu(self.conv3(x))   # → (batch, 64,  7,  7)
        x = x.flatten(1)            # → (batch, 3136)
        x = F.relu(self.fc1(x))     # → (batch, 512)
        x = self.fc2(x)             # → (batch, n_actions)
        return x

    def save(self, file_name: str = 'snake_cnn_model.pth'):
        folder = './model'
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, file_name)
        torch.save(self.state_dict(), path)
        print(f"[OK] CNN weights saved → {path}")

    def load(self, file_name: str = 'snake_cnn_model.pth', device: str = 'cpu'):
        path = os.path.join('./model', file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"[X] No checkpoint found at: {path}")
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        self.eval()
        print(f"[OK] CNN weights loaded ← {path}")


class ConvQTrainer:
    """
    Trainer for ConvQNet.  Handles the Bellman update and backprop.

    Key differences from the original QTrainer:
    - Input is a (B, 4, 84, 84) tensor rather than a 1D state vector.
    - Gradient clipping (max_norm=10) prevents exploding gradients,
      which are more common with pixel inputs.
    """

    def __init__(self, model: ConvQNet, lr: float = 1e-4, gamma: float = 0.99):
        self.model     = model
        self.gamma     = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        self.criterion = nn.SmoothL1Loss()   # Huber loss - more robust than MSE

    def train_step(self, state, action, reward, next_state, done):
        """
        One Bellman update step.
        
        state / next_state : numpy arrays of shape (4, 84, 84) or (B, 4, 84, 84)
        action             : one-hot list of length n_actions, or batch
        reward             : scalar or list
        done               : bool or list
        """
        state      = torch.tensor(np.array(state),      dtype=torch.float32)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32)
        action     = torch.tensor(np.array(action),     dtype=torch.long)
        reward     = torch.tensor(np.array(reward),     dtype=torch.float32)

        # Ensure batch dimension exists
        if state.dim() == 3:
            state      = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action     = action.unsqueeze(0)
            reward     = reward.unsqueeze(0)
            done       = (done,)

        # Current Q-values
        pred = self.model(state)               # (B, n_actions)

        # Bellman target
        target = pred.clone().detach()
        with torch.no_grad():
            next_q = self.model(next_state)    # (B, n_actions)

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(next_q[idx])
            act_idx = torch.argmax(action[idx]).item()
            target[idx][act_idx] = Q_new

        # Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()


# ── Quick architecture sanity check ───────────────────────────
if __name__ == '__main__':
    print("\n─── Snake ConvQNet Architecture ───")
    model = ConvQNet(n_actions=3)
    dummy = torch.zeros(1, FRAME_STACK, FRAME_H, FRAME_W)
    out   = model(dummy)
    print(f"Input  : {tuple(dummy.shape)}")
    print(f"Output : {tuple(out.shape)}  (Q-values for 3 actions)")
    total = sum(p.numel() for p in model.parameters())
    print(f"Params : {total:,}")
    print()
    print(model)
