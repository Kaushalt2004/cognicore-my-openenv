"""
CogniCore Pygame Renderer — Visual 2D rendering for RL environments.

Provides real-time visualization of MazeRunner, GridWorld, and Survival.
Supports render_mode="human" (window) and render_mode="rgb_array" (for video).

Usage:
    env = gym.make("cognicore/MazeRunner-v0", render_mode="human")
    obs, info = env.reset()
    env.render()  # Opens pygame window
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Any

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


# Color palette — premium dark theme
COLORS = {
    "bg":       (18, 18, 24),
    "wall":     (45, 45, 60),
    "floor":    (30, 30, 42),
    "agent":    (0, 200, 150),
    "agent_glow": (0, 255, 180),
    "goal":     (255, 200, 0),
    "goal_glow": (255, 230, 80),
    "trap":     (220, 50, 50),
    "visited":  (40, 50, 70),
    "path":     (60, 80, 120),
    "text":     (200, 200, 220),
    "text_dim": (120, 120, 140),
    "health_good": (0, 200, 100),
    "health_mid":  (255, 180, 0),
    "health_bad":  (220, 50, 50),
    "panel":    (25, 25, 35),
    "border":   (60, 60, 80),
    "grid_line": (35, 35, 50),
}

CELL_SIZE = 40
PANEL_WIDTH = 200
HEADER_HEIGHT = 50


class MazeRenderer:
    """Renders MazeRunner environment with pygame."""

    def __init__(self, size: int, render_mode: str = "human"):
        if not HAS_PYGAME:
            raise ImportError("pip install pygame")

        self.size = size
        self.render_mode = render_mode
        self.cell_size = min(CELL_SIZE, 600 // size)
        self.grid_w = size * self.cell_size
        self.grid_h = size * self.cell_size
        self.width = self.grid_w + PANEL_WIDTH
        self.height = self.grid_h + HEADER_HEIGHT

        self.screen = None
        self.clock = None
        self.font = None
        self.font_small = None
        self._initialized = False
        self.frame_count = 0

    def _init_pygame(self):
        if self._initialized:
            return
        pygame.init()
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("CogniCore — MazeRunner")
        else:
            self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self.font_small = pygame.font.SysFont("consolas", 12)
        self._initialized = True

    def render(
        self,
        walls, agent_pos, goal_pos, visit_counts,
        steps, max_steps, trap_memory=None,
    ) -> Optional[np.ndarray]:
        self._init_pygame()
        self.frame_count += 1

        # Handle pygame events
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

        # Clear
        self.screen.fill(COLORS["bg"])

        # Draw header
        self._draw_header(steps, max_steps)

        # Draw grid
        y_offset = HEADER_HEIGHT
        for r in range(self.size):
            for c in range(self.size):
                x = c * self.cell_size
                y = r * self.cell_size + y_offset
                pos = (r, c)
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)

                if pos in walls:
                    pygame.draw.rect(self.screen, COLORS["wall"], rect)
                elif pos in visit_counts:
                    # Visited cells get brighter with more visits
                    intensity = min(visit_counts[pos] * 15, 80)
                    color = (30 + intensity, 30 + intensity // 2, 42 + intensity)
                    pygame.draw.rect(self.screen, color, rect)
                else:
                    pygame.draw.rect(self.screen, COLORS["floor"], rect)

                # Grid lines
                pygame.draw.rect(self.screen, COLORS["grid_line"], rect, 1)

        # Draw goal (pulsing glow)
        gx = goal_pos[1] * self.cell_size
        gy = goal_pos[0] * self.cell_size + y_offset
        pulse = abs((self.frame_count % 40) - 20) / 20.0
        goal_color = tuple(
            int(COLORS["goal"][i] + (COLORS["goal_glow"][i] - COLORS["goal"][i]) * pulse)
            for i in range(3)
        )
        goal_rect = pygame.Rect(gx + 4, gy + 4, self.cell_size - 8, self.cell_size - 8)
        pygame.draw.rect(self.screen, goal_color, goal_rect, border_radius=4)
        # Goal label
        g_text = self.font_small.render("G", True, COLORS["bg"])
        self.screen.blit(g_text, (gx + self.cell_size // 2 - 4, gy + self.cell_size // 2 - 6))

        # Draw agent (pulsing)
        ax = agent_pos[1] * self.cell_size
        ay = agent_pos[0] * self.cell_size + y_offset
        agent_color = tuple(
            int(COLORS["agent"][i] + (COLORS["agent_glow"][i] - COLORS["agent"][i]) * pulse)
            for i in range(3)
        )
        agent_rect = pygame.Rect(ax + 3, ay + 3, self.cell_size - 6, self.cell_size - 6)
        pygame.draw.rect(self.screen, agent_color, agent_rect, border_radius=6)
        a_text = self.font_small.render("A", True, COLORS["bg"])
        self.screen.blit(a_text, (ax + self.cell_size // 2 - 4, ay + self.cell_size // 2 - 6))

        # Draw panel
        self._draw_panel(agent_pos, goal_pos, visit_counts, steps, max_steps, trap_memory)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(10)
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), (1, 0, 2)
            )

    def _draw_header(self, steps, max_steps):
        header = pygame.Rect(0, 0, self.width, HEADER_HEIGHT)
        pygame.draw.rect(self.screen, COLORS["panel"], header)
        pygame.draw.line(self.screen, COLORS["border"], (0, HEADER_HEIGHT),
                        (self.width, HEADER_HEIGHT))

        title = self.font.render("CogniCore — MazeRunner", True, COLORS["agent"])
        self.screen.blit(title, (10, 8))

        step_text = self.font_small.render(
            f"Step {steps}/{max_steps}", True, COLORS["text_dim"]
        )
        self.screen.blit(step_text, (10, 30))

        # Progress bar
        bar_x = 150
        bar_w = self.grid_w - 160
        bar_h = 8
        bar_y = 34
        pygame.draw.rect(self.screen, COLORS["wall"],
                        pygame.Rect(bar_x, bar_y, bar_w, bar_h), border_radius=4)
        fill = int(bar_w * min(1.0, steps / max(max_steps, 1)))
        if fill > 0:
            pygame.draw.rect(self.screen, COLORS["agent"],
                            pygame.Rect(bar_x, bar_y, fill, bar_h), border_radius=4)

    def _draw_panel(self, agent_pos, goal_pos, visit_counts, steps, max_steps, trap_memory):
        px = self.grid_w
        panel = pygame.Rect(px, 0, PANEL_WIDTH, self.height)
        pygame.draw.rect(self.screen, COLORS["panel"], panel)
        pygame.draw.line(self.screen, COLORS["border"], (px, 0), (px, self.height))

        y = HEADER_HEIGHT + 10

        def draw_stat(label, value, color=COLORS["text"]):
            nonlocal y
            lbl = self.font_small.render(label, True, COLORS["text_dim"])
            val = self.font.render(str(value), True, color)
            self.screen.blit(lbl, (px + 10, y))
            self.screen.blit(val, (px + 10, y + 14))
            y += 38

        dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        draw_stat("Distance", f"{dist} cells", COLORS["goal"])
        draw_stat("Position", f"({agent_pos[0]},{agent_pos[1]})")
        draw_stat("Explored", f"{len(visit_counts)} cells", COLORS["agent"])
        draw_stat("Steps", f"{steps}/{max_steps}")

        walls_known = len(trap_memory) if trap_memory else 0
        draw_stat("Walls Hit", f"{walls_known}", COLORS["trap"] if walls_known else COLORS["text"])

        efficiency = 1.0 - steps / max(max_steps, 1)
        draw_stat("Efficiency", f"{efficiency:.0%}",
                 COLORS["health_good"] if efficiency > 0.5 else COLORS["health_bad"])

    def close(self):
        if self._initialized:
            pygame.quit()
            self._initialized = False


class SurvivalRenderer:
    """Renders Survival environment status bars."""

    def __init__(self, render_mode: str = "human"):
        if not HAS_PYGAME:
            raise ImportError("pip install pygame")
        self.render_mode = render_mode
        self.width = 500
        self.height = 350
        self.screen = None
        self.clock = None
        self.font = None
        self._initialized = False

    def _init_pygame(self):
        if self._initialized:
            return
        pygame.init()
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("CogniCore — Survival")
        else:
            self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self.font_small = pygame.font.SysFont("consolas", 12)
        self._initialized = True

    def render(self, health, hunger, shelter, tools, food, materials,
               threat, weather, day, alive) -> Optional[np.ndarray]:
        self._init_pygame()

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

        self.screen.fill(COLORS["bg"])

        # Title
        title = self.font.render(f"Survival — Day {day}", True, COLORS["agent"])
        self.screen.blit(title, (10, 10))

        if not alive:
            dead = self.font.render("DEAD", True, COLORS["trap"])
            self.screen.blit(dead, (self.width - 60, 10))

        # Status bars
        y = 45
        bars = [
            ("Health", health, 100, COLORS["health_good"] if health > 50 else COLORS["health_bad"]),
            ("Hunger", hunger, 100, COLORS["health_bad"] if hunger > 60 else COLORS["health_mid"]),
            ("Food", food, 20, COLORS["goal"]),
            ("Materials", materials, 20, COLORS["text"]),
            ("Shelter", shelter, 5, COLORS["agent"]),
            ("Tools", tools, 5, COLORS["agent"]),
            ("Threat", threat, 5, COLORS["trap"]),
        ]

        for label, value, max_val, color in bars:
            lbl = self.font_small.render(f"{label}: {value:.0f}", True, COLORS["text_dim"])
            self.screen.blit(lbl, (10, y))

            bar_x, bar_w, bar_h = 120, 350, 16
            pygame.draw.rect(self.screen, COLORS["wall"],
                            pygame.Rect(bar_x, y, bar_w, bar_h), border_radius=4)
            fill = int(bar_w * min(1.0, value / max(max_val, 1)))
            if fill > 0:
                pygame.draw.rect(self.screen, color,
                                pygame.Rect(bar_x, y, fill, bar_h), border_radius=4)
            y += 30

        # Weather
        weather_names = ["☀ Clear", "🌧 Rain", "⛈ Storm", "🔥 Heat"]
        w_name = weather_names[int(weather) % 4]
        w_text = self.font.render(f"Weather: {w_name}", True, COLORS["text"])
        self.screen.blit(w_text, (10, y + 10))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(4)
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), (1, 0, 2)
            )

    def close(self):
        if self._initialized:
            pygame.quit()
            self._initialized = False
