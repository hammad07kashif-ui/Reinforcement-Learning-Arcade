import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
# FIX: Use SysFont to avoid "File Not Found" errors
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# --- NEON / DARK DESIGN COLORS ---
BG_COLOR = (25, 25, 40)       
GRID_COLOR_1 = (25, 25, 40)   # Darker square
GRID_COLOR_2 = (30, 30, 50)   # Lighter square

SNAKE_COLOR = (50, 255, 80)   # Neon Green
FOOD_COLOR = (255, 60, 60)    # Bright Red
WHITE = (255, 255, 255)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI - Training Mode')
        self.clock = pygame.time.Clock()
        self.reset()

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
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
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
        
        # --- UPDATED HUNGER LOGIC ---
        # WAS: 100 * len(snake). 
        # NOW: 40 * len(snake). 
        # If it spins in circles, it will starve much faster.
        if self.is_collision() or self.frame_iteration > 40*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            self.frame_iteration = 0 
        else:
            self.snake.pop()
        
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        # Draw Background
        self.display.fill(BG_COLOR)
        for x in range(0, self.w, BLOCK_SIZE):
            for y in range(0, self.h, BLOCK_SIZE):
                rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
                if (x // BLOCK_SIZE + y // BLOCK_SIZE) % 2 == 0:
                    pygame.draw.rect(self.display, GRID_COLOR_2, rect)
                else:
                    pygame.draw.rect(self.display, GRID_COLOR_1, rect)

        # Draw Snake (Neon Green)
        for pt in self.snake:
            pygame.draw.rect(self.display, SNAKE_COLOR, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BG_COLOR, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE), 1)

        # Draw Food
        pygame.draw.rect(self.display, FOOD_COLOR, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("SCORE: " + str(self.score), True, WHITE)
        self.display.blit(text, [10, 5])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else: 
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)