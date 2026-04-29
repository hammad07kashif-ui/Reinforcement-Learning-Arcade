import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
import math

pygame.init()
font = pygame.font.SysFont('arial', 25)

# --- NEON AESTHETIC ---
BG_COLOR = (10, 10, 20)
PADDLE_LEFT_COLOR = (0, 255, 255)   # Cyan
PADDLE_RIGHT_COLOR = (255, 0, 255)  # Magenta
BALL_COLOR = (255, 255, 255)
LINE_COLOR = (40, 40, 60)

BLOCK_SIZE = 20
SPEED = 200  # Fast speed for rapid AI training

class PongGameAI:
    def __init__(self, w=800, h=600):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Neural Arcade - Neon Pong Self-Play')
        self.clock = pygame.time.Clock()
        
        # Dimensions
        self.paddle_w = 15
        self.paddle_h = 100
        self.ball_size = 15
        
        # Game State
        self.score_left = 0
        self.score_right = 0
        self.frame_iteration = 0
        
        # Glow trail
        self.ball_trail = []
        
        self.reset()

    def reset(self):
        # Paddle positions (center Y)
        self.paddle_left_y = self.h // 2 - self.paddle_h // 2
        self.paddle_right_y = self.h // 2 - self.paddle_h // 2
        
        self.paddle_speed = 10
        
        # Ball position (center)
        self.ball_x = self.w // 2 - self.ball_size // 2
        self.ball_y = self.h // 2 - self.ball_size // 2
        
        # Ball Velocity
        self.base_ball_speed = 7.0
        angle = random.uniform(-math.pi/4, math.pi/4)
        direction = 1 if random.random() < 0.5 else -1
        
        self.ball_vx = self.base_ball_speed * math.cos(angle) * direction
        self.ball_vy = self.base_ball_speed * math.sin(angle)
        
        self.ball_trail.clear()
        self.frame_iteration = 0

    def play_step(self, action_left, action_right):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 1. Move paddles
        self._move_paddle(action_left, is_left=True)
        self._move_paddle(action_right, is_left=False)
        
        # 2. Move ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        
        # Update trail
        self.ball_trail.append((self.ball_x, self.ball_y))
        if len(self.ball_trail) > 10:
            self.ball_trail.pop(0)
            
        reward_left = 0
        reward_right = 0
        done = False
        
        # 3. Collision with top/bottom walls
        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_vy *= -1
        elif self.ball_y >= self.h - self.ball_size:
            self.ball_y = self.h - self.ball_size
            self.ball_vy *= -1
            
        # Rects for collision
        ball_rect = pygame.Rect(self.ball_x, self.ball_y, self.ball_size, self.ball_size)
        left_rect = pygame.Rect(30, self.paddle_left_y, self.paddle_w, self.paddle_h)
        right_rect = pygame.Rect(self.w - 30 - self.paddle_w, self.paddle_right_y, self.paddle_w, self.paddle_h)
        
        # 4. Collision with paddles
        if ball_rect.colliderect(left_rect) and self.ball_vx < 0:
            self.ball_x = 30 + self.paddle_w
            self.ball_vx *= -1.05 # Increase speed slightly
            self._adjust_angle(self.paddle_left_y)
            reward_left = 1  # AI Dopamine for hitting
            
        elif ball_rect.colliderect(right_rect) and self.ball_vx > 0:
            self.ball_x = self.w - 30 - self.paddle_w - self.ball_size
            self.ball_vx *= -1.05
            self._adjust_angle(self.paddle_right_y)
            reward_right = 1 # AI Dopamine for hitting
            
        # 5. Scoring (Left/Right boundaries)
        if self.ball_x < 0:
            self.score_right += 1
            reward_left = -10
            reward_right = 10
            done = True
        elif self.ball_x > self.w:
            self.score_left += 1
            reward_left = 10
            reward_right = -10
            done = True
            
        # Cap max frames to prevent infinite rallying during self-play
        if self.frame_iteration > 2000:
            done = True
            
        # 6. Update UI
        self._update_ui()
        self.clock.tick(SPEED)
        
        return reward_left, reward_right, done, self.score_left, self.score_right

    def _adjust_angle(self, paddle_y):
        # Bounce angle depends on where it hits the paddle
        relative_intersect_y = (paddle_y + (self.paddle_h / 2)) - (self.ball_y + (self.ball_size / 2))
        normalized_intersect = (relative_intersect_y / (self.paddle_h / 2))
        bounce_angle = normalized_intersect * (math.pi / 3) # Max 60 degrees
        
        speed = math.hypot(self.ball_vx, self.ball_vy)
        direction = 1 if self.ball_vx > 0 else -1
        
        self.ball_vx = speed * math.cos(bounce_angle) * direction
        self.ball_vy = speed * -math.sin(bounce_angle)

    def _move_paddle(self, action, is_left):
        # [stay, up, down]
        y = self.paddle_left_y if is_left else self.paddle_right_y
        
        if np.array_equal(action, [0, 1, 0]): # UP
            y -= self.paddle_speed
        elif np.array_equal(action, [0, 0, 1]): # DOWN
            y += self.paddle_speed
            
        # Bound to screen
        y = max(0, min(self.h - self.paddle_h, y))
        
        if is_left:
            self.paddle_left_y = y
        else:
            self.paddle_right_y = y

    def _update_ui(self):
        self.display.fill(BG_COLOR)
        
        # Center dashed line
        for y in range(0, self.h, 40):
            pygame.draw.rect(self.display, LINE_COLOR, (self.w//2 - 2, y, 4, 20))
            
        # Ball trail
        for i, (tx, ty) in enumerate(self.ball_trail):
            alpha = int(255 * (i / len(self.ball_trail)))
            size = max(1, int(self.ball_size * (i / len(self.ball_trail))))
            color = (BALL_COLOR[0], BALL_COLOR[1], BALL_COLOR[2])
            # Draw faded trail circles
            surface = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.ellipse(surface, (*color, alpha), (0, 0, size, size))
            self.display.blit(surface, (tx + self.ball_size//2 - size//2, ty + self.ball_size//2 - size//2))
            
        # Paddles (Neon glow approximation by drawing multiple layered rects)
        self._draw_neon_rect(30, self.paddle_left_y, self.paddle_w, self.paddle_h, PADDLE_LEFT_COLOR)
        self._draw_neon_rect(self.w - 30 - self.paddle_w, self.paddle_right_y, self.paddle_w, self.paddle_h, PADDLE_RIGHT_COLOR)
        
        # Ball
        pygame.draw.ellipse(self.display, BALL_COLOR, (self.ball_x, self.ball_y, self.ball_size, self.ball_size))
        
        # Score
        score_text = font.render(f"{self.score_left}   {self.score_right}", True, (200, 200, 200))
        self.display.blit(score_text, (self.w // 2 - score_text.get_width() // 2, 20))
        
        pygame.display.flip()
        
    def _draw_neon_rect(self, x, y, w, h, color):
        pygame.draw.rect(self.display, color, (x, y, w, h))
        # Inner white core
        pygame.draw.rect(self.display, (255, 255, 255), (x+4, y+4, w-8, h-8))
