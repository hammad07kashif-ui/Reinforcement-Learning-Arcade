import torch
import random
import numpy as np
import os
from collections import deque
from pacman_game import PacmanGameAI
from pacman_model import Linear_QNet, QTrainer
from flappy_helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class PacmanAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.95
        self.memory = deque(maxlen=MAX_MEMORY)
        
        # 12 State inputs
        self.model = Linear_QNet(12, 256, 4) 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        if os.path.exists('./model/pacman_model.pth'):
            try:
                self.model.load_state_dict(torch.load('./model/pacman_model.pth'))
                self.model.eval()
                print("[OK] Pacman Brain Loaded.")
            except:
                print("Starting fresh Pacman brain.")

    def get_state(self, game):
        px, py = game.pacman_pos
        
        # Directions: UP, DOWN, LEFT, RIGHT
        dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        walls = [0.0, 0.0, 0.0, 0.0]
        ghosts_dist = [0.0, 0.0, 0.0, 0.0]
        pellets_dist = [0.0, 0.0, 0.0, 0.0]
        
        # 1. Immediate walls
        for i, (dx, dy) in enumerate(dirs):
            nx, ny = px + dx, py + dy
            if nx < 0: nx = game.grid_w - 1
            elif nx >= game.grid_w: nx = 0
            if game.grid[ny][nx] == 1:
                walls[i] = 1.0
                
        # 2. Raycast for Ghosts and Pellets
        max_dist = float(max(game.grid_w, game.grid_h))
        
        for i, (dx, dy) in enumerate(dirs):
            dist = 1
            cx, cy = px + dx, py + dy
            
            while 0 <= cx < game.grid_w and 0 <= cy < game.grid_h and game.grid[cy][cx] != 1:
                # Check for ghost
                for ghost in game.ghosts:
                    if ghost['pos'][0] == cx and ghost['pos'][1] == cy:
                        # Closer ghost = higher threat (closer to 1.0)
                        if ghosts_dist[i] == 0.0: # Only care about the first one we see
                            ghosts_dist[i] = 1.0 - (dist / max_dist)
                            
                # Check for pellet
                if (cx, cy) in game.pellets:
                    if pellets_dist[i] == 0.0:
                        pellets_dist[i] = 1.0 - (dist / max_dist)
                        
                cx += dx
                cy += dy
                dist += 1
                
        state = walls + ghosts_dist + pellets_dist
        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
        else:
            mini_sample = self.memory
            
        if len(mini_sample) > 0:
            states, actions, rewards, next_states, dones = zip(*mini_sample)
            self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        if self.epsilon < 0:
            self.epsilon = 0
            
        final_move = [0, 0, 0, 0] # [UP, DOWN, LEFT, RIGHT]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    
    agent = PacmanAgent()
    game = PacmanGameAI()
    
    print("=" * 50)
    print(" STARTING PAC-MAN TRAINING ")
    print("=" * 50)

    while True:
        state_old = agent.get_state(game)
        action = agent.get_action(state_old)
        
        reward, done, score = game.play_step(action)
        
        state_new = agent.get_state(game)
        
        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)
        
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
                
            total_score += score
            mean_score = total_score / agent.n_games
            
            plot_scores.append(score)
            plot_mean_scores.append(mean_score)
            
            print(f'Game {agent.n_games} | Score: {score} | Record: {record}')
            
            try:
                plot(plot_scores, plot_mean_scores)
            except Exception:
                pass

if __name__ == '__main__':
    train()
