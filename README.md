# Multi-Game Reinforcement Learning Agent 🧠🎮

An autonomous Artificial Intelligence agent that learns to play classic arcade games using a **Deep Q-Network (DQN)**. Built entirely in Python and PyTorch, this AI starts with zero knowledge of the game and learns optimal pathfinding and decision-making purely through trial, error, and reinforcement learning.

Currently, the agent has mastered **Snake**, with a modular architecture designed to easily swap in new environments (like Flappy Bird) in the future.

## ✨ Key Features

* **Deep Q-Learning:** Utilizes a feed-forward neural network to predict the best possible moves based on the current state of the game.
* **Optimized State Space:** Uses a refined 11-input array (danger detection, current direction, food location) to maximize learning speed and efficiency.
* **Smart Memory & State Management:** Solves "catastrophic forgetting" by automatically saving the highest-performing neural weights (`model.pth`) and reloading them for future sessions with 0% randomness.
* **Custom Reward Shaping:** Implements "Hunger Logic" to penalize infinite looping, forcing the agent to take calculated risks to survive rather than playing overly safe.
* **Modular Architecture:** The core AI logic (`agent.py`, `model.py`) is completely decoupled from the game physics (`snake_game.py`), allowing for rapid deployment into new game environments.

## 🛠️ Tech Stack

* **Python 3.x**
* **PyTorch** (Deep Learning Framework)
* **Pygame** (Game Environment & Rendering)
* **NumPy** (Matrix Math & State representation)
* **Matplotlib** (Live Training Analytics)

## 📂 Project Structure

```text
GAME_AGENT/
│
├── agent.py            # The main RL agent logic (Memory, Training Steps, Action Selection)
├── model.py            # PyTorch Neural Network (Linear_QNet) and QTrainer
├── snake_game.py       # The Pygame environment and physics
├── helper.py           # Real-time plotting for scores and accuracy
└── model/              # Directory for saved brains
    └── model.pth       # The saved weights of the best-performing AI
