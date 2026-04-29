"""
╔══════════════════════════════════════════════════════════════╗
║         NEURAL ARCADE - SNAKE AI  |  INFERENCE MODE         ║
║  Loads pre-trained .pth weights. Epsilon locked to 0.        ║
║  No training, no memory, no replay. Pure exploitation.       ║
╚══════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn.functional as F
import numpy as np
import pygame
import sys
import os

from snake_game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from snake_model import Linear_QNet

# ── Model Config ───────────────────────────────────────────────
MODEL_PATH   = './model/snake_model.pth'
INPUT_SIZE   = 11
HIDDEN_SIZE  = 256
OUTPUT_SIZE  = 3
GAME_SPEED   = 18     # Snake steps per second (sweet spot for visual clarity)

# ── HUD Palette ────────────────────────────────────────────────
HUD_BG        = (10,  12,  28, 210)
ACCENT_GREEN  = (72,  255, 145)
ACCENT_BLUE   = (100, 180, 255)
ACCENT_RED    = (255, 90,  90)
ACCENT_GOLD   = (255, 215, 80)
TEXT_PRIMARY  = (230, 235, 255)
TEXT_DIM      = (120, 130, 160)
PANEL_BORDER  = (60,  70,  110)

ACTION_LABELS = ["STRAIGHT", "TURN R", "TURN L"]
ACTION_COLORS = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED]


# ══════════════════════════════════════════════════════════════
#  INFERENCE AGENT  -  Zero exploration, pure exploitation
# ══════════════════════════════════════════════════════════════
class SnakeInferenceAgent:
    def __init__(self):
        self.model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

        if not os.path.exists(MODEL_PATH):
            print(f"\n[X] Weights not found at: {MODEL_PATH}")
            print("[!] Run snake_agent.py first to train the model.\n")
            sys.exit(1)

        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        )
        self.model.eval()
        print("\n[OK] SNAKE MASTER BRAIN LOADED - Pure Exploitation Mode (eps = 0)\n")

    def get_state(self, game):
        head    = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight / right / left
            (dir_r and game.is_collision(point_r)) or (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or (dir_l and game.is_collision(point_d)),

            # Current direction
            dir_l, dir_r, dir_u, dir_d,

            # Food relative position
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]
        return np.array(state, dtype=int)

    @torch.no_grad()
    def get_action_with_confidence(self, state):
        """Returns (one-hot action, softmax probabilities, raw Q-values)."""
        state_t  = torch.tensor(state, dtype=torch.float)
        q_values = self.model(state_t)
        probs    = F.softmax(q_values, dim=0).numpy()
        move_idx = torch.argmax(q_values).item()
        action   = [0, 0, 0]
        action[move_idx] = 1
        return action, probs, q_values.numpy()


# ══════════════════════════════════════════════════════════════
#  PREMIUM HUD RENDERER
# ══════════════════════════════════════════════════════════════
class HUDRenderer:
    def __init__(self, display, game_w, game_h):
        self.display  = display
        self.game_w   = game_w
        self.game_h   = game_h

        # Fonts
        pygame.font.init()
        self.font_lg  = pygame.font.SysFont('Consolas', 18, bold=True)
        self.font_md  = pygame.font.SysFont('Consolas', 14, bold=True)
        self.font_sm  = pygame.font.SysFont('Consolas', 12)

        # Overlay surface (bottom strip)
        self.panel_h  = 110
        self.panel    = pygame.Surface((game_w, self.panel_h), pygame.SRCALPHA)
        self.pulse    = 0.0

    def _draw_bar(self, surface, x, y, w, h, fill, color, label, pct_text):
        """Draw a labelled progress bar."""
        pygame.draw.rect(surface, (30, 35, 60), (x, y, w, h), border_radius=4)
        fill_w = max(0, int(w * fill))
        if fill_w > 0:
            pygame.draw.rect(surface, color, (x, y, fill_w, h), border_radius=4)
        pygame.draw.rect(surface, PANEL_BORDER, (x, y, w, h), 1, border_radius=4)
        lbl = self.font_sm.render(label, True, TEXT_DIM)
        surface.blit(lbl, (x, y - 15))
        pct = self.font_sm.render(pct_text, True, TEXT_PRIMARY)
        surface.blit(pct, (x + w + 4, y))

    def render(self, score, record, n_games, probs, chosen_idx, fps):
        self.pulse = (self.pulse + 0.05) % (2 * 3.14159)
        alpha_pulse = int(180 + 50 * abs(np.sin(self.pulse)))

        panel = pygame.Surface((self.game_w, self.panel_h), pygame.SRCALPHA)
        panel.fill((8, 10, 24, 210))
        pygame.draw.line(panel, PANEL_BORDER, (0, 0), (self.game_w, 0), 2)

        # ── Left block: stats ──────────────────────────────────
        y0 = 10
        badge = self.font_lg.render("[!] INFERENCE MODE", True, ACCENT_GREEN)
        panel.blit(badge, (12, y0))

        stats = [
            (f"SCORE   {score:>4}", ACCENT_GOLD),
            (f"RECORD  {record:>4}", ACCENT_BLUE),
            (f"GAME #  {n_games:>4}", TEXT_PRIMARY),
            (f"FPS     {int(fps):>4}", TEXT_DIM),
        ]
        for i, (txt, col) in enumerate(stats):
            surf = self.font_md.render(txt, True, col)
            panel.blit(surf, (12, y0 + 24 + i * 18))

        # ── Right block: Q-value confidence bars ───────────────
        bar_x0 = self.game_w // 2 + 10
        bar_w  = self.game_w - bar_x0 - 20
        bar_h  = 13

        conf_lbl = self.font_md.render("NETWORK CONFIDENCE", True, TEXT_DIM)
        panel.blit(conf_lbl, (bar_x0, y0))

        for i, (prob, label, col) in enumerate(zip(probs, ACTION_LABELS, ACTION_COLORS)):
            bar_y = y0 + 22 + i * 26
            bdr_col = ACCENT_GREEN if i == chosen_idx else col
            self._draw_bar(panel, bar_x0, bar_y + 12, bar_w, bar_h,
                           prob, bdr_col, label, f"{prob*100:.1f}%")
            if i == chosen_idx:
                arrow = self.font_sm.render("< CHOSEN", True, ACCENT_GREEN)
                panel.blit(arrow, (bar_x0 - 60, bar_y + 12))

        # ── Epsilon indicator ──────────────────────────────────
        eps_surf = self.font_sm.render("eps = 0.000  |  MASTER EXPLOITATION", True, (alpha_pulse, 255, 120))
        panel.blit(eps_surf, (12, self.panel_h - 18))

        # Blit HUD BELOW the game area (game is 0..game_h, HUD is game_h..game_h+panel_h)
        self.display.blit(panel, (0, self.game_h))


# ══════════════════════════════════════════════════════════════
#  MAIN SHOWCASE LOOP
# ══════════════════════════════════════════════════════════════
def run():
    agent      = SnakeInferenceAgent()
    game_h_ext = 110   # Extra height for HUD panel
    game       = SnakeGameAI(w=640, h=480)

    # Resize window to include HUD strip
    display = pygame.display.set_mode((640, 480 + game_h_ext))
    pygame.display.set_caption("Neural Arcade - Snake AI  |  Inference Mode")
    game.display = display

    hud       = HUDRenderer(display, 640, 480)
    record    = 0
    n_games   = 0
    clock     = pygame.time.Clock()

    print("=" * 60)
    print("  SNAKE MASTER - PURE INFERENCE")
    print("  Close the window or press ESC to exit.")
    print("=" * 60)

    while True:
        # ── Event handling (allow ESC to quit) ─────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()

        state              = agent.get_state(game)
        action, probs, _   = agent.get_action_with_confidence(state)
        chosen_idx         = action.index(1)

        reward, done, score = game.play_step(action)

        # Render HUD below the game area and present the final frame
        fps = clock.get_fps()
        hud.render(score, record, n_games, probs, chosen_idx, fps)
        pygame.display.flip()   # Final authoritative flip (HUD included)
        clock.tick(GAME_SPEED)  # Cap to GAME_SPEED - game's internal tick is faster (40)

        if done:
            game.reset()
            n_games += 1
            if score > record:
                record = score
            print(f"  Game {n_games:>4}  |  Score {score:>3}  |  Record {record:>3}")


if __name__ == '__main__':
    run()
