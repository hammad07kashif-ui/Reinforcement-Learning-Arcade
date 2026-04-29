import pygame
import random
import numpy as np
import math

pygame.init()
font = pygame.font.SysFont('arial', 25)

# --- ARCADE AESTHETIC ---
BG_COLOR = (0, 0, 0)
WALL_COLOR = (25, 25, 166)     # Deep Arcade Blue
PELLET_COLOR = (255, 184, 174) # Peach
PACMAN_COLOR = (255, 255, 0)   # Yellow
GHOST_COLORS = [
    (255, 0, 0),    # Blinky (Red)
    (255, 184, 255),# Pinky (Pink)
    (0, 255, 255),  # Inky (Cyan)
    (255, 184, 82)  # Clyde (Orange)
]

CELL_SIZE = 40
SPEED = 120  # Fast speed for rapid AI training

RAW_MAP = [
    "WWWWWWWWWWWWWWW",
    "W......W......W",
    "W.WW.WWW.WW.W.W",
    "W.............W",
    "W.WW.W.WWW.WW.W",
    "W....W.000.W..W",
    "WWWW.W.0G0.W.WW",
    ".......0G0.....",
    "WWWW.W.000.W.WW",
    "W....W..P..W..W",
    "W.WW.W.WWW.WW.W",
    "W.............W",
    "W.WW.WWW.WW.W.W",
    "W......W......W",
    "WWWWWWWWWWWWWWW"
]

class PacmanGameAI:
    def __init__(self, w=600, h=650):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Neural Arcade - Pac-Man AI')
        self.clock = pygame.time.Clock()
        
        self.grid_w = len(RAW_MAP[0])
        self.grid_h = len(RAW_MAP)
        
        self.reset()

    def reset(self):
        self.grid = []
        self.pellets = set()
        self.ghosts = []
        self.pacman_pos = [0, 0]
        
        self.score = 0
        self.frame_iteration = 0
        
        # Parse map
        ghost_idx = 0
        for y, row in enumerate(RAW_MAP):
            grid_row = []
            for x, char in enumerate(row):
                if char == 'W':
                    grid_row.append(1) # Wall
                else:
                    grid_row.append(0) # Empty/Path
                    if char == '.':
                        self.pellets.add((x, y))
                    elif char == 'P':
                        self.pacman_pos = [x, y]
                    elif char == 'G':
                        self.ghosts.append({'pos': [x, y], 'color': GHOST_COLORS[ghost_idx % len(GHOST_COLORS)]})
                        ghost_idx += 1
            self.grid.append(grid_row)
            
        self.initial_pellets = len(self.pellets)

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        # Action: [UP, DOWN, LEFT, RIGHT]
        dx, dy = 0, 0
        if np.array_equal(action, [1, 0, 0, 0]): dy = -1
        elif np.array_equal(action, [0, 1, 0, 0]): dy = 1
        elif np.array_equal(action, [0, 0, 1, 0]): dx = -1
        elif np.array_equal(action, [0, 0, 0, 1]): dx = 1
        
        reward = 0
        done = False
        
        # 1. Move Pacman
        nx, ny = self.pacman_pos[0] + dx, self.pacman_pos[1] + dy
        
        # Wrapping logic for tunnel
        if nx < 0: nx = self.grid_w - 1
        elif nx >= self.grid_w: nx = 0
        
        hit_wall = False
        if self.grid[ny][nx] == 1:
            hit_wall = True
            reward = -1 # Penalty for hitting wall
        else:
            self.pacman_pos = [nx, ny]
            
        # 2. Check Pellets
        pos_tuple = tuple(self.pacman_pos)
        if pos_tuple in self.pellets:
            self.pellets.remove(pos_tuple)
            self.score += 1
            reward = 10
            
            if len(self.pellets) == 0:
                reward = 50
                done = True
        
        # 3. Move Ghosts
        for ghost in self.ghosts:
            # Simple AI: Random valid move
            valid_moves = []
            for gdx, gdy in [(0,-1), (0,1), (-1,0), (1,0)]:
                gx, gy = ghost['pos'][0] + gdx, ghost['pos'][1] + gdy
                if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                    if self.grid[gy][gx] != 1:
                        valid_moves.append((gx, gy))
            if valid_moves:
                # 20% chance to move towards pacman to make it challenging
                if random.random() < 0.2:
                    best_move = valid_moves[0]
                    min_dist = float('inf')
                    for vx, vy in valid_moves:
                        dist = abs(vx - self.pacman_pos[0]) + abs(vy - self.pacman_pos[1])
                        if dist < min_dist:
                            min_dist = dist
                            best_move = (vx, vy)
                    ghost['pos'] = list(best_move)
                else:
                    ghost['pos'] = list(random.choice(valid_moves))
                    
        # 4. Check Ghost Collision
        for ghost in self.ghosts:
            if ghost['pos'] == self.pacman_pos:
                reward = -50
                done = True
                
        # Timeout
        if self.frame_iteration > 1500:
            done = True
            
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, done, self.score

    def _update_ui(self):
        self.display.fill(BG_COLOR)
        
        # Draw Maze
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                px, py = x * CELL_SIZE, y * CELL_SIZE
                if self.grid[y][x] == 1:
                    # Draw Neon Wall
                    pygame.draw.rect(self.display, WALL_COLOR, (px + 4, py + 4, CELL_SIZE - 8, CELL_SIZE - 8), border_radius=5)
                    pygame.draw.rect(self.display, (0,0,255), (px + 4, py + 4, CELL_SIZE - 8, CELL_SIZE - 8), 1, border_radius=5)
                    
        # Draw Pellets
        for (x, y) in self.pellets:
            px, py = x * CELL_SIZE + CELL_SIZE//2, y * CELL_SIZE + CELL_SIZE//2
            pygame.draw.circle(self.display, PELLET_COLOR, (px, py), 4)
            
        # Draw Pacman
        px, py = self.pacman_pos[0] * CELL_SIZE + CELL_SIZE//2, self.pacman_pos[1] * CELL_SIZE + CELL_SIZE//2
        # Simple chomping animation
        if (self.frame_iteration // 5) % 2 == 0:
            pygame.draw.circle(self.display, PACMAN_COLOR, (px, py), CELL_SIZE//2 - 4)
        else:
            pygame.draw.circle(self.display, PACMAN_COLOR, (px, py), CELL_SIZE//2 - 4)
            pygame.draw.polygon(self.display, BG_COLOR, [(px, py), (px + CELL_SIZE, py - CELL_SIZE//2), (px + CELL_SIZE, py + CELL_SIZE//2)])
            
        # Draw Ghosts
        for ghost in self.ghosts:
            gx, gy = ghost['pos'][0] * CELL_SIZE + CELL_SIZE//2, ghost['pos'][1] * CELL_SIZE + CELL_SIZE//2
            color = ghost['color']
            # Draw ghost body (half circle + rect)
            pygame.draw.circle(self.display, color, (gx, gy - 4), CELL_SIZE//2 - 6)
            pygame.draw.rect(self.display, color, (gx - (CELL_SIZE//2 - 6), gy - 4, CELL_SIZE - 12, CELL_SIZE//2))
            
            # Draw eyes
            pygame.draw.circle(self.display, (255,255,255), (gx - 5, gy - 6), 4)
            pygame.draw.circle(self.display, (255,255,255), (gx + 5, gy - 6), 4)
            pygame.draw.circle(self.display, (0,0,255), (gx - 5, gy - 6), 2)
            pygame.draw.circle(self.display, (0,0,255), (gx + 5, gy - 6), 2)

        # Score Area at bottom
        text = font.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.display.blit(text, (10, self.h - 40))
        
        pygame.display.flip()
