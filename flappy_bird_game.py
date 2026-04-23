import pygame
import random
import numpy as np
import math
import sys
import os

pygame.init()
font = pygame.font.SysFont('arial', 32, bold=True)

WHITE = (255, 255, 255)
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800
BASE_HEIGHT = 700 
SPEED = 60 

class FlappyBirdAI:
    def __init__(self):
        self.w = SCREEN_WIDTH
        self.h = SCREEN_HEIGHT
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Flappy Bird AI - Coin Hunter')
        self.clock = pygame.time.Clock()
        
        self.pipe_width = 80    
        self.bird_width = 55     
        self.bird_height = 40    
        self.coin_size = 35
        
        self.bg_x = 0
        self.base_x = 0
        self.bg_speed = 2          
        self.base_speed = 7        
        
        self.bird_index = 0
        self._load_assets()
        self.reset()

    def _load_assets(self):
        self.bg_img = pygame.Surface((self.w, self.h))
        self.bg_img.fill((135, 206, 235)) 
        self.base_img = pygame.Surface((self.w, self.h - BASE_HEIGHT))
        self.base_img.fill((222, 216, 149)) 
        self.pipe_img = pygame.Surface((self.pipe_width, self.h))
        self.pipe_img.fill((34, 139, 34)) 
        self.pipe_top_img = self.pipe_img.copy()
        self.bird_imgs = [pygame.Surface((self.bird_width, self.bird_height)) for _ in range(3)]
        for img in self.bird_imgs: img.fill((255, 215, 0)) 
        self.coin_img = pygame.Surface((self.coin_size, self.coin_size))
        self.coin_img.fill((255, 223, 0)) 

        try:
            bg = pygame.image.load(os.path.join('flappy_assets', 'background.png')).convert()
            self.bg_img = pygame.transform.scale(bg, (self.w, self.h))
            base = pygame.image.load(os.path.join('flappy_assets', 'base.png')).convert()
            self.base_img = pygame.transform.scale(base, (self.w, self.h - BASE_HEIGHT))
            pipe_raw = pygame.image.load(os.path.join('flappy_assets', 'pipe.png')).convert_alpha()
            self.pipe_img = pygame.transform.scale(pipe_raw, (self.pipe_width, self.h))
            self.pipe_top_img = pygame.transform.flip(self.pipe_img, False, True)
            b1 = pygame.image.load(os.path.join('flappy_assets', 'frame-1.png')).convert_alpha()
            b2 = pygame.image.load(os.path.join('flappy_assets', 'frame-2.png')).convert_alpha()
            self.bird_imgs = [pygame.transform.scale(img, (self.bird_width, self.bird_height)) for img in [b1, b2, b1]]
            coin_raw = pygame.image.load(os.path.join('flappy_assets', 'coin.png')).convert_alpha()
            self.coin_img = pygame.transform.scale(coin_raw, (self.coin_size, self.coin_size))
        except Exception:
            pass

    def reset(self):
        self.bird_y = self.h // 2
        self.bird_velocity = 0
        self.gravity = 1.2
        self.jump_strength = -10
        
        self.pipe_x = self.w
        self.pipe_gap_y = random.randint(150, BASE_HEIGHT - 250)
        self.pipe_gap_height = 200
        
        self.score = 0
        self.coins_collected = 0
        self.frame_iteration = 0
        
        self.pipe_passed = False
        self.coin_active = True
        self.coin_x = self.pipe_x + (self.pipe_width // 2) - (self.coin_size // 2)
        self.coin_y = self.pipe_gap_y + (self.pipe_gap_height // 2) - (self.coin_size // 2)

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if np.array_equal(action, [0, 1]):
            self.bird_velocity = self.jump_strength

        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity

        self.pipe_x -= self.base_speed
        self.bg_x -= self.bg_speed
        self.base_x -= self.base_speed
        
        if self.bg_x <= -self.w: self.bg_x = 0
        if self.base_x <= -self.w: self.base_x = 0
        if self.coin_active: self.coin_x -= self.base_speed

        reward = 0.1 
        game_over = False

        bird_rect = pygame.Rect(100, self.bird_y, self.bird_width, self.bird_height)
        pipe_top_rect = pygame.Rect(self.pipe_x, 0, self.pipe_width, self.pipe_gap_y)
        pipe_bottom_rect = pygame.Rect(self.pipe_x, self.pipe_gap_y + self.pipe_gap_height, self.pipe_width, self.h)
        base_rect = pygame.Rect(0, BASE_HEIGHT, self.w, self.h - BASE_HEIGHT)

        # Fatal Collisions
        if bird_rect.colliderect(pipe_top_rect) or bird_rect.colliderect(pipe_bottom_rect) or bird_rect.colliderect(base_rect) or self.bird_y < 0:
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # EXPLICIT COIN REWARD LOGIC
        if self.coin_active:
            coin_rect = pygame.Rect(self.coin_x, self.coin_y, self.coin_size, self.coin_size)
            if bird_rect.colliderect(coin_rect):
                self.coin_active = False
                self.coins_collected += 1
                self.score += 2   # Boost the game score
                reward = 20       # MASSIVE AI Dopamine hit for getting a coin!

        # Pipe pass mechanism
        if self.pipe_x + self.pipe_width < 100 and not self.pipe_passed:
            self.score += 1
            reward = 10
            self.pipe_passed = True
            self.coin_active = False

        # Regenerate pipe
        if self.pipe_x < -self.pipe_width:
            self.pipe_x = self.w
            self.pipe_gap_y = random.randint(150, BASE_HEIGHT - 250)
            self.pipe_passed = False
            self.coin_active = True
            self.coin_x = self.pipe_x + (self.pipe_width // 2) - (self.coin_size // 2)
            self.coin_y = self.pipe_gap_y + (self.pipe_gap_height // 2) - (self.coin_size // 2)

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def _update_ui(self):
        self.display.blit(self.bg_img, (self.bg_x, 0))
        self.display.blit(self.bg_img, (self.bg_x + self.w, 0))
        self.display.blit(self.pipe_top_img, (self.pipe_x, self.pipe_gap_y - self.h))
        self.display.blit(self.pipe_img, (self.pipe_x, self.pipe_gap_y + self.pipe_gap_height))
        self.display.blit(self.base_img, (self.base_x, BASE_HEIGHT))
        self.display.blit(self.base_img, (self.base_x + self.w, BASE_HEIGHT))

        if self.coin_active:
            self.display.blit(self.coin_img, (self.coin_x, self.coin_y))

        self.bird_index = (self.bird_index + 1) % 30
        bird_img = self.bird_imgs[self.bird_index // 10]
        
        angle = -self.bird_velocity * 2
        angle = max(min(angle, 30), -90)
        rotated_bird = pygame.transform.rotate(bird_img, angle)
        self.display.blit(rotated_bird, (100, self.bird_y))

        text = font.render(f"Score: {self.score} | Coins: {self.coins_collected}", True, WHITE)
        self.display.blit(text, [10, 10])
        pygame.display.flip()