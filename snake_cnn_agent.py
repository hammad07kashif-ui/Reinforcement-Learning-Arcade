"""
╔══════════════════════════════════════════════════════════════╗
║    NEURAL ARCADE - SNAKE CNN AGENT  |  VISION UPGRADE (M2)   ║
║                                                              ║
║  The AI no longer receives coordinate data.                  ║
║  It "sees" the raw game pixels, just like a human would.     ║
║                                                              ║
║  Frame pipeline:                                             ║
║    Pygame Surface → Grayscale → 84x84 → Stack 4 → CNN        ║
╚══════════════════════════════════════════════════════════════╝

HOW IT WORKS:
  1. After every game step, we capture the pygame display surface.
  2. The RGB frame is converted to grayscale (luminance formula).
  3. It is downsampled to 84x84 pixels (PIL BILINEAR).
  4. We maintain a deque of the last 4 frames as the "state."
     This gives the CNN temporal context (it can infer motion/direction).
  5. The 4x84x84 tensor is fed to ConvQNet → Q-values → action.

WHY THIS IS BETTER:
  - The vector agent knows exactly where food is (coordinate cheat).
  - The CNN agent must LEARN what food looks like from pixel patterns.
  - This is closer to how humans perceive the game.
  - Transferable to ANY visual environment without re-engineering state.
"""

import torch
import random
import numpy as np
import pygame
import os
import time
from collections import deque
from PIL import Image

from snake_game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from snake_cnn_model import ConvQNet, ConvQTrainer
from snake_helper import plot

# ── Hyperparameters ────────────────────────────────────────────
MAX_MEMORY   = 50_000    # Reduced vs vector agent (frames use more RAM)
BATCH_SIZE   = 32        # Smaller batch for CNN stability
LR           = 1e-4      # Standard DQN learning rate
GAMMA        = 0.99
FRAME_STACK  = 4
FRAME_H      = 84
FRAME_W      = 84

# ── Exploration schedule ───────────────────────────────────────
EPS_START    = 1.0       # 100% random at start
EPS_END      = 0.05      # Floor at 5% random
EPS_DECAY    = 0.995     # Multiply epsilon by this each game


# ══════════════════════════════════════════════════════════════
#  FRAME PREPROCESSOR
# ══════════════════════════════════════════════════════════════
def preprocess_frame(surface: pygame.Surface) -> np.ndarray:
    """
    Convert a pygame Surface → (84, 84) float32 numpy array in [0,1].

    Steps:
      1. surfarray.array3d → (W, H, 3) uint8 RGB
      2. Transpose        → (H, W, 3)
      3. Grayscale        → (H, W) using luminance coefficients
      4. PIL resize       → (84, 84)
      5. Normalise        → divide by 255.0
    """
    rgb = pygame.surfarray.array3d(surface)   # (W, H, 3) uint8
    rgb = rgb.transpose(1, 0, 2)              # (H, W, 3)

    # Luminance-weighted grayscale (ITU-R BT.601)
    gray = (0.299 * rgb[:, :, 0] +
            0.587 * rgb[:, :, 1] +
            0.114 * rgb[:, :, 2]).astype(np.uint8)

    # Resize to 84x84 with PIL (high-quality BILINEAR)
    pil_img  = Image.fromarray(gray).resize((FRAME_W, FRAME_H), Image.BILINEAR)
    frame    = np.array(pil_img, dtype=np.float32) / 255.0   # [0, 1]
    return frame   # shape: (84, 84)


# ══════════════════════════════════════════════════════════════
#  FRAME STACK  -  Maintains temporal context for the CNN
# ══════════════════════════════════════════════════════════════
class FrameStack:
    """
    Maintains a rolling window of the last `k` processed frames.

    reset()         : Fill the stack with zeros (used on game reset).
    push(frame)     : Add a new frame; oldest is discarded.
    get_state()     : Returns (k, H, W) numpy array ready for the CNN.
    """

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
        """Returns shape (4, 84, 84) - ready to be passed to ConvQNet."""
        return np.stack(list(self.frames), axis=0)


# ══════════════════════════════════════════════════════════════
#  CNN AGENT
# ══════════════════════════════════════════════════════════════
class SnakeCNNAgent:
    def __init__(self):
        self.n_games     = 0
        self.epsilon     = EPS_START
        self.gamma       = GAMMA
        self.memory      = deque(maxlen=MAX_MEMORY)

        self.model       = ConvQNet(n_actions=3)
        self.trainer     = ConvQTrainer(self.model, lr=LR, gamma=GAMMA)
        self.frame_stack = FrameStack(k=FRAME_STACK)

        # Resume training if checkpoint exists
        ckpt = './model/snake_cnn_model.pth'
        if os.path.exists(ckpt):
            try:
                self.model.load(device='cpu')
                print("[OK] Resuming CNN training from checkpoint.")
            except Exception as e:
                print(f"[!] Could not load checkpoint: {e}  - starting fresh.")

    # ── Exploration (eps-greedy, exponential decay) ──────────────
    def get_action(self, state: np.ndarray):
        """
        eps-greedy policy.
        Returns (one-hot action list, chosen index)
        """
        final_move = [0, 0, 0]

        if random.random() < self.epsilon:
            move = random.randint(0, 2)
        else:
            state_t     = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_t)
            move = torch.argmax(q_values).item()

        final_move[move] = 1
        return final_move, move

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    # ── Memory ─────────────────────────────────────────────────
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
    total_score      = 0
    record           = 0

    agent = SnakeCNNAgent()
    game  = SnakeGameAI()

    print("\n" + "=" * 65)
    print("  SNAKE CNN AGENT - VISION MODE TRAINING")
    print("  The AI learns from raw pixels. No coordinate cheating.")
    print(f"  eps starts at {EPS_START:.2f} → decays to {EPS_END:.2f} over games.")
    print("=" * 65 + "\n")

    while True:
        # ── Get pixel state (4 stacked frames) ─────────────────
        frame = preprocess_frame(game.display)
        agent.frame_stack.push(frame)
        state_old = agent.frame_stack.get_state()   # (4, 84, 84)

        # ── Action ─────────────────────────────────────────────
        action, _ = agent.get_action(state_old)

        # ── Step ───────────────────────────────────────────────
        reward, done, score = game.play_step(action)

        # ── New pixel state ────────────────────────────────────
        frame_new = preprocess_frame(game.display)
        agent.frame_stack.push(frame_new)
        state_new = agent.frame_stack.get_state()

        # ── Short-memory train ─────────────────────────────────
        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            game.reset()
            agent.frame_stack.reset()
            agent.n_games += 1
            agent.decay_epsilon()

            # ── Long-memory train ──────────────────────────────
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
                
                # Export to ONNX for web deployment
                dummy_input = torch.zeros(1, FRAME_STACK, FRAME_H, FRAME_W, dtype=torch.float32)
                onnx_path = os.path.join('./model', 'snake_cnn_model.onnx')
                try:
                    torch.onnx.export(
                        agent.model, 
                        dummy_input, 
                        onnx_path, 
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=['input'], 
                        output_names=['output'],
                        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                    )
                    print(f"[OK] ONNX model exported → {onnx_path}")
                except Exception as e:
                    print(f"[!] ONNX export failed: {e}")

            total_score      += score
            mean_score        = total_score / agent.n_games
            plot_scores.append(score)
            plot_mean_scores.append(mean_score)

            print(
                f"  Game {agent.n_games:>4}  |  Score {score:>3}  |  "
                f"Record {record:>3}  |  eps {agent.epsilon:.4f}  |  "
                f"Mean {mean_score:.2f}"
            )

            try:
                plot(plot_scores, plot_mean_scores, [])
            except Exception:
                pass


if __name__ == '__main__':
    train()
