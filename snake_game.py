import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import os

pygame.init()
pygame.mixer.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# --- UI COLORS ---
BG_COLOR = (25, 25, 40)       
GRID_COLOR_1 = (25, 25, 40)   
GRID_COLOR_2 = (30, 30, 50)   
WHITE = (255, 255, 255)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI - Fixed Sprite Logic')
        self.clock = pygame.time.Clock()
        
        self._load_assets()
        self.reset()

    def _load_assets(self):
        self.sprites = {}
        sprite_names = [
            'head_up', 'head_down', 'head_left', 'head_right',
            'body_vertical', 'body_horizontal',
            'curve_top_left', 'curve_top_right', 'curve_bottom_left', 'curve_bottom_right',
            'tail_up', 'tail_down', 'tail_left', 'tail_right',
            'food'
        ]
        
        for name in sprite_names:
            try:
                path = os.path.join('snake_assets', f'{name}.png')
                img = pygame.image.load(path).convert_alpha()
                self.sprites[name] = pygame.transform.scale(img, (BLOCK_SIZE, BLOCK_SIZE))
            except Exception:
                surf = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE))
                surf.fill((0, 200, 50))
                self.sprites[name] = surf

        # Sounds
        self.eat_sound = None
        self.die_sound = None
        try:
            self.eat_sound = pygame.mixer.Sound(os.path.join('snake_assets', 'EatSound_CC0_by_EugeneLoza.ogg'))
            self.die_sound = pygame.mixer.Sound(os.path.join('snake_assets', 'DieSound_CC0_by_EugeneLoza.ogg'))
        except:
            pass

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        self._move(action) 
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            if self.die_sound: self.die_sound.play()
            return reward, game_over, self.score
            
        if self.head == self.food:
            self.score += 1
            reward = 10
            if self.eat_sound: self.eat_sound.play()
            self._place_food()
        else:
            self.snake.pop()
        
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None: pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BG_COLOR)

        # 1. Background Grid
        for y in range(0, self.h, BLOCK_SIZE):
            for x in range(0, self.w, BLOCK_SIZE):
                rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
                color = GRID_COLOR_1 if (x // BLOCK_SIZE + y // BLOCK_SIZE) % 2 == 0 else GRID_COLOR_2
                pygame.draw.rect(self.display, color, rect)

        # 2. Draw Snake
        for i, pt in enumerate(self.snake):
            # --- HEAD ---
            if i == 0:
                if self.direction == Direction.UP: img = self.sprites['head_up']
                elif self.direction == Direction.DOWN: img = self.sprites['head_down']
                elif self.direction == Direction.LEFT: img = self.sprites['head_left']
                else: img = self.sprites['head_right']
            
            # --- TAIL ---
            elif i == len(self.snake) - 1:
                # Tail points toward the segment in front of it
                next_seg = self.snake[i-1]
                if next_seg.x < pt.x: img = self.sprites['tail_left']
                elif next_seg.x > pt.x: img = self.sprites['tail_right']
                elif next_seg.y < pt.y: img = self.sprites['tail_up']
                else: img = self.sprites['tail_down']

            # --- BODY / CURVES ---
            else:
                prev_seg = self.snake[i-1] # Toward Head
                next_seg = self.snake[i+1] # Toward Tail

                # Check if it's a straight line
                if prev_seg.x == next_seg.x:
                    img = self.sprites['body_vertical']
                elif prev_seg.y == next_seg.y:
                    img = self.sprites['body_horizontal']
                
                # It's a curve! Calculate relative vectors
                else:
                    # p = previous segment relative to current, n = next segment relative
                    px, py = prev_seg.x - pt.x, prev_seg.y - pt.y
                    nx, ny = next_seg.x - pt.x, next_seg.y - pt.y

                    # Logic to find which of the 4 corners we are in
                    if (px == -BLOCK_SIZE or nx == -BLOCK_SIZE) and (py == -BLOCK_SIZE or ny == -BLOCK_SIZE):
                        img = self.sprites['curve_top_left']
                    elif (px == BLOCK_SIZE or nx == BLOCK_SIZE) and (py == -BLOCK_SIZE or ny == -BLOCK_SIZE):
                        img = self.sprites['curve_top_right']
                    elif (px == -BLOCK_SIZE or nx == -BLOCK_SIZE) and (py == BLOCK_SIZE or ny == BLOCK_SIZE):
                        img = self.sprites['curve_bottom_left']
                    else:
                        img = self.sprites['curve_bottom_right']

            self.display.blit(img, (pt.x, pt.y))

        # 3. Food & Score
        self.display.blit(self.sprites['food'], (self.food.x, self.food.y))
        text = font.render(f"SCORE: {self.score}", True, WHITE)
        self.display.blit(text, [10, 5])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] 
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4] 
        else: 
            new_dir = clock_wise[(idx - 1) % 4] 

        self.direction = new_dir
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction == Direction.LEFT: x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN: y += BLOCK_SIZE
        elif self.direction == Direction.UP: y -= BLOCK_SIZE
        self.head = Point(x, y)