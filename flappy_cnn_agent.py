"""
╔══════════════════════════════════════════════════════════════╗
║  NEURAL ARCADE - FLAPPY CNN AGENT  |  VISION UPGRADE (M2)    ║
║                                                              ║
║  The Flappy Bird AI learns to navigate pipes from pixels.    ║
║  Frame pipeline:                                             ║
║    Pygame Surface → Grayscale → 84x84 → Stack 4 → CNN        ║
╚══════════════════════════════════════════════════════════════╝

DESIGN NOTES for Flappy CNN vs Snake CNN:
  - Flappy Bird has more temporal dependency than Snake.
    The bird's velocity cannot be inferred from a single frame -
    you need at least 2 frames to see if it's rising or falling.
    4-frame stacking gives the CNN full motion awareness.

  - The reward signal is unchanged (existing flappy_bird_game.py).
    Only the STATE representation changes: pixel frames instead of scalars.

  - Flappy is harder to pixel-train than Snake because:
    (a) the pipe gap is a subtle visual feature at 84x84 resolution
    (b) the scrolling background adds visual noise
    Tip: Increase initial eps exploration to ~1.0 and train for 500+ games.
"""

import torch
import random
import numpy as np
import pygame
import os
from collections import deque
from PIL import Image

from flappy_bird_game import FlappyBirdAI
from flappy_cnn_model import ConvQNet, ConvQTrainer
from flappy_helper import plot

# ── Hyperparameters ────────────────────────────────────────────
MAX_MEMORY   = 50_000
BATCH_SIZE   = 32
LR           = 1e-4
GAMMA        = 0.99
FRAME_STACK  = 4
FRAME_H      = 84
FRAME_W      = 84

EPS_START    = 1.0
EPS_END      = 0.05
EPS_DECAY    = 0.995


# ══════════════════════════════════════════════════════════════
#  FRAME PREPROCESSOR
# ══════════════════════════════════════════════════════════════
def preprocess_frame(surface: pygame.Surface) -> np.ndarray:
    """
    Flappy-specific preprocessing.
    
    Same pipeline as Snake, but note:
    - The game window is 600x800 (portrait) - tall and narrow.
    - We resize to 84x84 which squashes the aspect ratio.
      This is fine - the CNN learns invariant features regardless.
    - The base (ground) strip is a prominent horizontal feature
      the CNN will naturally learn as a danger boundary.
    """
    rgb  = pygame.surfarray.array3d(surface)   # (W, H, 3) uint8
    rgb  = rgb.transpose(1, 0, 2)              # (H, W, 3)

    gray = (0.299 * rgb[:, :, 0] +
            0.587 * rgb[:, :, 1] +
            0.114 * rgb[:, :, 2]).astype(np.uint8)

    pil_img = Image.fromarray(gray).resize((FRAME_W, FRAME_H), Image.BILINEAR)
    return np.array(pil_img, dtype=np.float32) / 255.0   # (84, 84) ∈ [0,1]


# ══════════════════════════════════════════════════════════════
#  FRAME STACK
# ══════════════════════════════════════════════════════════════
class FrameStack:
    def __init__(self, k: int = FRAME_STACK):
        self.k      = k
        self.frames = deque(maxlen=k)
        self.reset()

    def reset(self):
        for _ in range(self.k):
            self.frames.append(np.zeros((FRAME_H, FRAME_W), dtype=np.float32))

    def push(self, frame: np.ndarray):
        self.frames.append(frame)

    def get_state(self) -> np.ndarray:
        return np.stack(list(self.frames), axis=0)   # (4, 84, 84)


# ══════════════════════════════════════════════════════════════
#  CNN AGENT
# ══════════════════════════════════════════════════════════════
class FlappyCNNAgent:
    def __init__(self):
        self.n_games     = 0
        self.epsilon     = EPS_START
        self.memory      = deque(maxlen=MAX_MEMORY)

        self.model       = ConvQNet(n_actions=2)
        self.trainer     = ConvQTrainer(self.model, lr=LR, gamma=GAMMA)
        self.frame_stack = FrameStack(k=FRAME_STACK)

        self.record      = 0
        self.total_score = 0

        ckpt = './model/flappy_cnn_model.pth'
        if os.path.exists(ckpt):
            try:
                self.model.load(device='cpu')
                print("[OK] Resuming Flappy CNN training from checkpoint.")
            except Exception as e:
                print(f"[!] Could not load checkpoint: {e}  - starting fresh.")

    def get_action(self, state: np.ndarray):
        """eps-greedy with Flappy-specific bias: prefer glide (0) during exploration."""
        final_move = [0, 0]

        if random.random() < self.epsilon:
            # Flappy-tuned exploration: 85% glide, 15% flap
            # Prevents the agent from flapping itself into the ceiling early on
            move = 1 if random.random() < 0.15 else 0
        else:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.model(state_t)
            move = torch.argmax(q_vals).item()

        final_move[move] = 1
        return final_move, move

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


# ══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════
def train():
    plot_scores      = []
    plot_mean_scores = []

    agent = FlappyCNNAgent()
    game  = FlappyBirdAI()

    print("\n" + "=" * 65)
    print("  FLAPPY BIRD CNN AGENT - VISION MODE TRAINING")
    print("  The AI learns from raw pixels. No physics coordinate cheating.")
    print(f"  eps starts at {EPS_START:.2f} → decays to {EPS_END:.2f} over games.")
    print("  Tip: CNN Flappy needs ~300 games to show competence.")
    print("=" * 65 + "\n")

    while True:
        # ── Pixel state ────────────────────────────────────────
        frame = preprocess_frame(game.display)
        agent.frame_stack.push(frame)
        state_old = agent.frame_stack.get_state()

        action, _ = agent.get_action(state_old)

        reward, done, score = game.play_step(action)

        frame_new = preprocess_frame(game.display)
        agent.frame_stack.push(frame_new)
        state_new = agent.frame_stack.get_state()

        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            game.reset()
            agent.frame_stack.reset()
            agent.n_games += 1
            agent.decay_epsilon()

            agent.train_long_memory()

            if score > agent.record:
                agent.record = score
                agent.model.save()

            agent.total_score += score
            mean_score         = agent.total_score / agent.n_games
            plot_scores.append(score)
            plot_mean_scores.append(mean_score)

            print(
                f"  Game {agent.n_games:>4}  |  Score {score:>3}  |  "
                f"Record {agent.record:>3}  |  eps {agent.epsilon:.4f}  |  "
                f"Mean {mean_score:.2f}"
            )

            try:
                plot(plot_scores, plot_mean_scores)
            except Exception:
                pass


if __name__ == '__main__':
    train()
