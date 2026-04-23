import torch
import random
import numpy as np
import os
import json
from collections import deque
from flappy_bird_game import FlappyBirdAI
from flappy_model import Linear_QNet, QTrainer
from flappy_helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class FlappyAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 
        self.gamma = 0.9 
        self.memory = deque(maxlen=MAX_MEMORY) 
        
        # UPGRADED: 7 Inputs for Coin Vision!
        self.model = Linear_QNet(7, 256, 2) 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        self.record = 0
        self.plot_scores = []
        self.plot_mean_scores = []
        self.total_score = 0

        if os.path.exists('./model/flappy_model.pth'):
            try:
                self.model.load_state_dict(torch.load('./model/flappy_model.pth'))
                self.model.eval()
                print(">>> SUCCESS: FLAPPY BRAIN LOADED. <<<")
                
                if os.path.exists('./model/flappy_training_data.json'):
                    with open('./model/flappy_training_data.json', 'r') as f:
                        data = json.load(f)
                        self.n_games = data.get('n_games', 0)
                        self.record = data.get('record', 0)
                        self.plot_scores = data.get('plot_scores', [])
                        self.plot_mean_scores = data.get('plot_mean_scores', [])
                        self.total_score = data.get('total_score', 0)
            except:
                print(">>> ARCHITECTURE UPGRADED: Starting fresh Flappy brain. <<<")

    def save_stats(self):
        data = {
            'n_games': self.n_games,
            'record': self.record,
            'plot_scores': self.plot_scores,
            'plot_mean_scores': self.plot_mean_scores,
            'total_score': self.total_score
        }
        with open('./model/flappy_training_data.json', 'w') as f:
            json.dump(data, f)

    def get_state(self, game):
        # NORMALIZED INPUTS + COIN VISION
        state = [
            game.bird_y / game.h,                                                
            game.bird_velocity / 15.0,                                           
            (game.pipe_x - 100) / game.w,                                        
            (game.bird_y - game.pipe_gap_y) / game.h,                            
            ((game.pipe_gap_y + game.pipe_gap_height) - game.bird_y) / game.h,
            (game.coin_x - 100) / game.w,            
            (game.coin_y - game.bird_y) / game.h     
        ]
        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 100 - self.n_games
        if self.epsilon < 0: 
            self.epsilon = 0
            
        final_move = [0,0] 
        
        if random.randint(0, 200) < self.epsilon:
            # 90% chance to do nothing, 10% chance to jump during exploration
            if random.random() < 0.10: 
                final_move[1] = 1 
            else: 
                final_move[0] = 1 
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move

def train():
    agent = FlappyAgent()
    game = FlappyBirdAI()
    
    if len(agent.plot_scores) > 0:
        plot(agent.plot_scores, agent.plot_mean_scores)

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > agent.record:
                agent.record = score
                agent.model.save()

            agent.total_score += score
            agent.plot_scores.append(score)
            mean_score = agent.total_score / len(agent.plot_scores)
            agent.plot_mean_scores.append(mean_score)
            agent.save_stats()

            print('Game', agent.n_games, 'Score', score, 'Record:', agent.record)
            try: 
                plot(agent.plot_scores, agent.plot_mean_scores)
            except Exception: 
                pass

if __name__ == '__main__':
    train()