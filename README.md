# 🕹️ Neural Arcade AI

Welcome to the **Neural Arcade**, a premium portfolio showcase of Reinforcement Learning environments built from scratch. This project demonstrates the evolution from simple vector-based AI to DeepMind-style Convolutional Neural Networks (CNN) learning entirely from raw pixels.

## 🚀 Features & Milestones

*   **Milestone 1: Zero-Overhead Inference** — Custom `play_only` scripts load pre-trained `.pth` weights for maximum FPS, featuring custom UI telemetry HUDs to visualize the AI's Q-values and confidence in real-time.
*   **Milestone 2: Deep Convolutional Q-Networks (CNN)** — Upgraded agents that "see" the game. A custom frame pipeline converts Pygame surfaces to 84x84 grayscale tensors, stacks 4 temporal frames, and feeds them into a 1.6M parameter Deep Q-Network.
*   **Milestone 4: Complex Environments** — Built entirely custom implementations of Multi-Agent Pong (self-play shared brain) and Pac-Man (virtual raycasting) with premium neon graphics.
*   **Milestone 5: Web Deployment Ready** — PyTorch models are automatically exported to **ONNX** format during training, allowing seamless deployment to web browsers using ONNX Runtime Web without needing a Python backend.

---

## 🎮 The Games

### 1. 🐍 Snake AI
*   **Architecture**: CNN (Raw Pixel Vision) & Vector 
*   **Mechanics**: The AI navigates a grid, growing larger while avoiding its own tail.
*   **Inference**: Real-time Q-value bar charts injected directly into the Pygame window.

### 2. 🐦 Flappy Bird AI
*   **Architecture**: CNN (Raw Pixel Vision) & Vector 
*   **Mechanics**: A continuous physics environment. The AI must balance velocity and gravity to navigate scrolling pipes.
*   **Inference**: Live physics telemetry dashboard showing velocity, height, and pipe distance.

### 3. 🏓 Pong (Multi-Agent Self-Play)
*   **Architecture**: Vector (7-Parameter Relative State)
*   **Mechanics**: Two paddles controlled by a single "Shared Brain". The network mirrors its perspective depending on which side it's playing, halving training time.
*   **Aesthetics**: Pitch black with glowing Cyan/Magenta paddles and dynamic ball trails.

### 4. 🟡 Pac-Man (Raycast Maze)
*   **Architecture**: Vector (12-Parameter Virtual Raycast)
*   **Mechanics**: The AI shoots 12 virtual rays (Up, Down, Left, Right) to sense walls, pellets, and the distance to 4 hostile ghosts.
*   **Aesthetics**: Classic arcade blue glowing maze with colored ghost entities.

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/Reinforcement-Learning-Arcade.git
cd Reinforcement-Learning-Arcade
```

2. Install dependencies:
```bash
pip install torch pygame numpy matplotlib pillow onnx
```

---

## 🎯 Usage Guide

### Watch the Pre-Trained Masters (Inference Mode)
To see the AI play perfectly at high speed with beautiful UI overlays:
```bash
python snake_play_only.py
python flappy_play_only.py
```

### Train the DeepMind CNNs (Pixel Vision)
To watch the AI learn how to play simply by "looking" at the screen pixels:
```bash
python snake_cnn_agent.py
python flappy_cnn_agent.py
```

### Train the Advanced Environments
To train the self-play Pong brain or the maze-navigating Pac-Man:
```bash
python pong_agent.py
python pacman_agent.py
```

---

## 🧠 Technical Details

**CNN Architecture (ConvQNet):**
*   **Input Layer**: 4x84x84 Grayscale Temporal Frame Stack
*   **Hidden Layers**: 3 Convolutional Layers (Feature Extraction) + 2 Linear Layers
*   **Loss Function**: Huber Loss (Smooth L1) with Gradient Clipping (`max_norm=10`)
*   **Initialization**: Orthogonal weight initialization

**Training Parameters:**
*   **Learning Rate**: 0.001 (Adam Optimizer)
*   **Discount Factor (γ)**: 0.9 - 0.95
*   **Memory Buffer**: 100,000 experiences (Deque)
*   **Batch Size**: 1,000

## 📂 Project Structure

```text
Reinforcement-Learning-Arcade/
├── [Game].py                 # The Pygame engine/environment
├── [Game]_model.py           # PyTorch Neural Network definitions
├── [Game]_agent.py           # The Reinforcement Learning training loop
├── [Game]_play_only.py       # Standalone high-FPS inference scripts
├── flappy_assets/            # Audio & Visual assets
├── snake_assets/             # Audio & Visual assets
└── model/                    # Saved .pth weights
```