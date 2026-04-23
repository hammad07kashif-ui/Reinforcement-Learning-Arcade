# Reinforcement Learning Arcade

A collection of classic arcade games enhanced with AI agents trained using reinforcement learning techniques. This project implements Q-learning with neural networks to create intelligent agents that can play Flappy Bird and Snake games.

## Games Included

### 🐦 Flappy Bird AI - Coin Hunter
An enhanced version of the classic Flappy Bird game where the AI agent learns to navigate pipes while collecting coins for bonus rewards. The agent uses advanced state representation including coin vision to optimize its strategy.

### 🐍 Snake Game AI
A traditional Snake game where the AI agent learns to navigate the game board, avoid collisions, and maximize its score by eating food while growing longer.

## Features

- **Deep Q-Learning**: Both agents use neural networks trained with Q-learning algorithm
- **Experience Replay**: Memory buffers store game experiences for efficient training
- **Model Persistence**: Trained models are automatically saved and can be loaded for continued training or gameplay
- **Training Analytics**: Real-time plotting of scores and performance metrics
- **Epsilon-Greedy Exploration**: Balanced exploration vs exploitation strategy during training
- **Advanced State Representation**: Rich input features for better decision making

## Dependencies

- Python 3.7+
- PyTorch
- Pygame
- NumPy
- Matplotlib (for plotting)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Reinforcement-Learning-Arcade.git
cd Reinforcement-Learning-Arcade
```

2. Install required packages:
```bash
pip install torch pygame numpy matplotlib
```

## Usage

### Training the AI Agents

#### Flappy Bird Agent
```bash
python flappy_agent.py
```

#### Snake Agent
```bash
python snake_agent.py
```

### How It Works

1. **Initialization**: The agent loads any existing trained model or starts fresh
2. **Game Loop**: For each game episode:
   - Agent observes the current game state
   - Chooses an action using epsilon-greedy policy
   - Receives reward and observes next state
   - Updates Q-network using experience replay
3. **Model Saving**: Best performing models are automatically saved
4. **Visualization**: Training progress is plotted in real-time

### Game Controls (Manual Play)

Both games support manual play through keyboard input:
- **Flappy Bird**: Spacebar to jump
- **Snake**: Arrow keys to change direction

## Project Structure

```
Reinforcement-Learning-Arcade/
├── flappy_agent.py          # Flappy Bird AI agent with Q-learning
├── flappy_bird_game.py      # Flappy Bird game implementation
├── flappy_helper.py         # Helper functions for Flappy Bird
├── flappy_model.py          # Neural network model for Flappy Bird
├── snake_agent.py           # Snake AI agent with Q-learning
├── snake_game.py            # Snake game implementation
├── snake_helper.py          # Helper functions for Snake
├── snake_model.py           # Neural network model for Snake
├── flappy_assets/           # Flappy Bird game assets
├── snake_assets/            # Snake game assets
├── model/                   # Saved trained models and training data
│   ├── flappy_model.pth
│   ├── flappy_training_data.json
│   └── snake_model.pth
└── README.md
```

## Technical Details

### Neural Network Architecture
- **Input Layer**: Game state features (positions, velocities, directions)
- **Hidden Layer**: 256 neurons with ReLU activation
- **Output Layer**: Action probabilities (2 for Flappy Bird, 3 for Snake)

### Training Parameters
- **Learning Rate**: 0.001
- **Discount Factor (γ)**: 0.9
- **Memory Buffer**: 100,000 experiences
- **Batch Size**: 1,000
- **Epsilon Decay**: Starts at 100, decreases with games played

### Reward System
- **Flappy Bird**: +1 for passing pipes, +10 for collecting coins, -1000 for collision
- **Snake**: +10 for eating food, -10 for collision, +1 for survival bonus

## Model Files

The `model/` directory contains:
- `flappy_model.pth`: Trained neural network weights for Flappy Bird
- `snake_model.pth`: Trained neural network weights for Snake
- `flappy_training_data.json`: Training statistics and progress data



## Acknowledgments

- Inspired by classic arcade games
- Built using PyTorch for deep learning
- Pygame for game development
- Reinforcement learning concepts from various research papers