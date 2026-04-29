import torch
import random
import numpy as np
import os
from collections import deque
from pong_game import PongGameAI
from pong_model import Linear_QNet, QTrainer
from flappy_helper import plot # Re-using plotting logic

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class PongAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.95
        self.memory = deque(maxlen=MAX_MEMORY)
        
        # 7 State inputs
        self.model = Linear_QNet(7, 256, 3) 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        if os.path.exists('./model/pong_model.pth'):
            try:
                self.model.load_state_dict(torch.load('./model/pong_model.pth'))
                self.model.eval()
                print("[OK] Pong Shared Brain Loaded.")
            except:
                print("Starting fresh Pong brain.")

    def get_state(self, game, is_left):
        # We need a relative state so the same brain can play both sides.
        
        paddle_y = game.paddle_left_y if is_left else game.paddle_right_y
        opp_paddle_y = game.paddle_right_y if is_left else game.paddle_left_y
        
        if is_left:
            dist_x = game.ball_x - 30
            vx = game.ball_vx
            incoming = 1.0 if game.ball_vx < 0 else 0.0
        else:
            dist_x = (game.w - 30) - game.ball_x
            vx = -game.ball_vx # Relative velocity
            incoming = 1.0 if game.ball_vx > 0 else 0.0
            
        state = [
            paddle_y / game.h,
            dist_x / game.w,
            (game.ball_y - paddle_y) / game.h,
            vx / 15.0,
            game.ball_vy / 15.0,
            opp_paddle_y / game.h,
            incoming
        ]
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
            
        final_move = [0, 0, 0] # [STAY, UP, DOWN]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
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
    
    agent = PongAgent()
    game = PongGameAI()
    
    print("=" * 50)
    print(" STARTING PONG SELF-PLAY TRAINING ")
    print("=" * 50)

    while True:
        # Get old states for both
        state_old_l = agent.get_state(game, is_left=True)
        state_old_r = agent.get_state(game, is_left=False)
        
        # Get actions
        action_l = agent.get_action(state_old_l)
        action_r = agent.get_action(state_old_r)
        
        # Play step
        reward_l, reward_r, done, score_l, score_r = game.play_step(action_l, action_r)
        
        # Get new states
        state_new_l = agent.get_state(game, is_left=True)
        state_new_r = agent.get_state(game, is_left=False)
        
        # Train short memory for both
        agent.train_short_memory(state_old_l, action_l, reward_l, state_new_l, done)
        agent.train_short_memory(state_old_r, action_r, reward_r, state_new_r, done)
        
        # Remember for both
        agent.remember(state_old_l, action_l, reward_l, state_new_l, done)
        agent.remember(state_old_r, action_r, reward_r, state_new_r, done)
        
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            # Record is based on total rallies/points in a game to keep it saving
            total_points = score_l + score_r
            if total_points > record:
                record = total_points
                agent.model.save()
                
            total_score += total_points
            mean_score = total_score / agent.n_games
            
            plot_scores.append(total_points)
            plot_mean_scores.append(mean_score)
            
            print(f'Game {agent.n_games} | L: {score_l}  R: {score_r} | Record {record}')
            
            try:
                plot(plot_scores, plot_mean_scores)
            except Exception:
                pass

if __name__ == '__main__':
    train()
