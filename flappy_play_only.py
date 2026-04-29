"""
╔══════════════════════════════════════════════════════════════╗
║       NEURAL ARCADE - FLAPPY BIRD AI  |  INFERENCE MODE      ║
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

from flappy_bird_game import FlappyBirdAI
from flappy_model import Linear_QNet

# ── Model Config ───────────────────────────────────────────────
MODEL_PATH   = './model/flappy_model.pth'
INPUT_SIZE   = 7
HIDDEN_SIZE  = 256
OUTPUT_SIZE  = 2

# ── HUD Palette ────────────────────────────────────────────────
ACCENT_SKY    = (100, 200, 255)
ACCENT_GOLD   = (255, 215, 80)
ACCENT_GREEN  = (72,  255, 145)
ACCENT_RED    = (255, 90,  90)
ACCENT_ORANGE = (255, 160, 60)
TEXT_PRIMARY  = (230, 235, 255)
TEXT_DIM      = (120, 130, 160)
PANEL_BORDER  = (60,  70,  110)
HUD_BG_ALPHA  = 210

ACTION_LABELS  = ["GLIDE v", "FLAP ^"]
ACTION_COLORS  = [ACCENT_ORANGE, ACCENT_SKY]


# ══════════════════════════════════════════════════════════════
#  INFERENCE AGENT
# ══════════════════════════════════════════════════════════════
class FlappyInferenceAgent:
    def __init__(self):
        self.model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

        if not os.path.exists(MODEL_PATH):
            print(f"\n[X] Weights not found at: {MODEL_PATH}")
            print("[!] Run flappy_agent.py first to train the model.\n")
            sys.exit(1)

        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        )
        self.model.eval()
        print("\n[OK] FLAPPY MASTER BRAIN LOADED - Pure Exploitation Mode (eps = 0)\n")

    def get_state(self, game):
        state = [
            game.bird_y / game.h,
            game.bird_velocity / 15.0,
            (game.pipe_x - 100) / game.w,
            (game.bird_y - game.pipe_gap_y) / game.h,
            ((game.pipe_gap_y + game.pipe_gap_height) - game.bird_y) / game.h,
            (game.coin_x - 100) / game.w,
            (game.coin_y - game.bird_y) / game.h,
        ]
        return np.array(state, dtype=float)

    @torch.no_grad()
    def get_action_with_confidence(self, state):
        state_t  = torch.tensor(state, dtype=torch.float)
        q_values = self.model(state_t)
        probs    = F.softmax(q_values, dim=0).numpy()
        move_idx = torch.argmax(q_values).item()
        action   = [0, 0]
        action[move_idx] = 1
        return action, probs, q_values.numpy()


# ══════════════════════════════════════════════════════════════
#  PREMIUM HUD RENDERER  (side panel - overlaid on right edge)
# ══════════════════════════════════════════════════════════════
class FlappyHUDRenderer:
    def __init__(self, display, game_w, game_h):
        self.display = display
        self.game_w  = game_w
        self.game_h  = game_h
        self.panel_w = 200
        self.pulse   = 0.0

        pygame.font.init()
        self.font_lg = pygame.font.SysFont('Consolas', 16, bold=True)
        self.font_md = pygame.font.SysFont('Consolas', 13, bold=True)
        self.font_sm = pygame.font.SysFont('Consolas', 11)

    def _draw_bar(self, surface, x, y, w, h, fill, color, label, value_text):
        pygame.draw.rect(surface, (20, 25, 50), (x, y, w, h), border_radius=4)
        fw = max(0, int(w * fill))
        if fw > 0:
            pygame.draw.rect(surface, color, (x, y, fw, h), border_radius=4)
        pygame.draw.rect(surface, PANEL_BORDER, (x, y, w, h), 1, border_radius=4)
        lbl = self.font_sm.render(label, True, TEXT_DIM)
        surface.blit(lbl, (x, y - 14))
        val = self.font_sm.render(value_text, True, TEXT_PRIMARY)
        surface.blit(val, (x + w + 4, y - 2))

    def _vel_color(self, velocity):
        """Color shifts from sky-blue (ascending) to red (falling fast)."""
        t = min(max((velocity + 10) / 25.0, 0.0), 1.0)
        r = int(ACCENT_RED[0] * t + ACCENT_SKY[0] * (1 - t))
        g = int(ACCENT_RED[1] * t + ACCENT_SKY[1] * (1 - t))
        b = int(ACCENT_RED[2] * t + ACCENT_SKY[2] * (1 - t))
        return (r, g, b)

    def render(self, score, coins, record, n_games, probs, chosen_idx,
               fps, bird_y, bird_vel, pipe_dist, game_h):
        self.pulse = (self.pulse + 0.04) % (2 * 3.14159)
        pulse_alpha = int(170 + 55 * abs(np.sin(self.pulse)))

        pw   = self.panel_w
        ph   = self.game_h
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill((8, 10, 24, 200))
        pygame.draw.line(panel, PANEL_BORDER, (0, 0), (0, ph), 2)

        y = 14
        # ── Badge ──────────────────────────────────────────────
        badge = self.font_lg.render("[!] INFERENCE", True, ACCENT_GREEN)
        panel.blit(badge, (10, y)); y += 22
        sub = self.font_sm.render("eps = 0  |  MASTER MODE", True, (pulse_alpha, 255, 120))
        panel.blit(sub, (10, y)); y += 22

        pygame.draw.line(panel, PANEL_BORDER, (5, y), (pw - 5, y)); y += 10

        # ── Score block ────────────────────────────────────────
        stats = [
            ("SCORE",   f"{score}",   ACCENT_GOLD),
            ("COINS",   f"{coins}",   ACCENT_GOLD),
            ("RECORD",  f"{record}",  ACCENT_SKY),
            ("GAME #",  f"{n_games}", TEXT_PRIMARY),
            ("FPS",     f"{int(fps)}", TEXT_DIM),
        ]
        for label, val, col in stats:
            l_s = self.font_md.render(f"{label:<7}", True, TEXT_DIM)
            v_s = self.font_md.render(val, True, col)
            panel.blit(l_s, (10, y))
            panel.blit(v_s, (pw - 10 - v_s.get_width(), y))
            y += 18

        pygame.draw.line(panel, PANEL_BORDER, (5, y + 4), (pw - 5, y + 4)); y += 16

        # ── Physics telemetry ──────────────────────────────────
        tele = self.font_md.render("TELEMETRY", True, TEXT_DIM)
        panel.blit(tele, (10, y)); y += 18

        height_pct = 1.0 - (bird_y / game_h)
        vel_col     = self._vel_color(bird_vel)
        vel_norm    = min(max((bird_vel + 10) / 25.0, 0.0), 1.0)
        pipe_norm   = min(max(pipe_dist / 600.0, 0.0), 1.0)

        self._draw_bar(panel, 10, y + 14, pw - 20, 10,
                       height_pct, ACCENT_GREEN, "HEIGHT", f"{int(height_pct*100)}%")
        y += 32

        self._draw_bar(panel, 10, y + 14, pw - 20, 10,
                       vel_norm, vel_col, "VELOCITY", f"{bird_vel:.1f}")
        y += 32

        self._draw_bar(panel, 10, y + 14, pw - 20, 10,
                       pipe_norm, ACCENT_ORANGE, "PIPE DIST", f"{int(pipe_dist)}px")
        y += 36

        pygame.draw.line(panel, PANEL_BORDER, (5, y), (pw - 5, y)); y += 12

        # ── Action confidence bars ─────────────────────────────
        conf = self.font_md.render("CONFIDENCE", True, TEXT_DIM)
        panel.blit(conf, (10, y)); y += 18

        for i, (prob, label, col) in enumerate(zip(probs, ACTION_LABELS, ACTION_COLORS)):
            bdr = ACCENT_GREEN if i == chosen_idx else col
            self._draw_bar(panel, 10, y + 14, pw - 20, 12,
                           prob, bdr, label, f"{prob*100:.1f}%")
            if i == chosen_idx:
                chk = self.font_sm.render("<", True, ACCENT_GREEN)
                panel.blit(chk, (pw - 18, y + 14))
            y += 32

        self.display.blit(panel, (self.game_w - pw, 0))


# ══════════════════════════════════════════════════════════════
#  MAIN SHOWCASE LOOP
# ══════════════════════════════════════════════════════════════
def run():
    agent = FlappyInferenceAgent()

    # Widen window to accommodate side HUD panel
    PANEL_W   = 200
    GAME_W    = 600
    GAME_H    = 800
    WIN_W     = GAME_W + PANEL_W

    pygame.init()
    display = pygame.display.set_mode((WIN_W, GAME_H))
    pygame.display.set_caption("Neural Arcade - Flappy Bird AI  |  Inference Mode")

    # Create game AFTER we set up the display so it adopts our window
    game = FlappyBirdAI()
    game.display = display  # Redirect game rendering to our wider canvas
    game.w       = GAME_W   # Game logic stays within its 600px coordinate space

    hud      = FlappyHUDRenderer(display, GAME_W, GAME_H)
    record   = 0
    n_games  = 0
    clock    = pygame.time.Clock()

    print("=" * 60)
    print("  FLAPPY BIRD MASTER - PURE INFERENCE")
    print("  Close the window or press ESC to exit.")
    print("=" * 60)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()

        state              = agent.get_state(game)
        action, probs, _   = agent.get_action_with_confidence(state)
        chosen_idx         = action.index(1)

        reward, done, score = game.play_step(action)

        fps = clock.get_fps()
        pipe_dist = max(game.pipe_x - 100, 0)

        hud.render(
            score, game.coins_collected, record, n_games,
            probs, chosen_idx, fps,
            game.bird_y, game.bird_velocity, pipe_dist, GAME_H
        )
        pygame.display.flip()
        clock.tick(60)

        if done:
            # Capture coins BEFORE reset() zeroes them out
            coins_this_game = game.coins_collected
            game.reset()
            n_games += 1
            if score > record:
                record = score
            print(f"  Game {n_games:>4}  |  Score {score:>3}  |  Coins {coins_this_game}  |  Record {record:>3}")


if __name__ == '__main__':
    run()
