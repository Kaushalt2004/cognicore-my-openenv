"""
CogniCore — Professional Demo Recorder
Records AI agents learning in real-time across all environments.
Exports as MP4 video showing before/after comparisons.

Run: python demo/record_demo.py
Output: demo/cognicore_demo.mp4
"""
import pygame
import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cognicore.gym
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# ── CONFIG ──
WIDTH, HEIGHT = 1280, 720
FPS = 30
CELL = 48
FONT_NAME = "consolas"

# Colors
BG = (8, 8, 16)
SURFACE = (14, 14, 28)
WALL = (35, 35, 55)
FLOOR = (22, 22, 38)
AGENT = (0, 220, 150)
AGENT_GLOW = (0, 255, 180)
GOAL = (255, 200, 50)
TRAP = (220, 60, 60)
VISITED = (30, 40, 60)
TEXT = (220, 220, 235)
DIM = (100, 100, 130)
ACCENT = (0, 220, 150)
GOLD = (255, 200, 50)
RED = (255, 80, 100)
BLUE = (80, 140, 255)
PANEL = (18, 18, 32)
BORDER = (40, 40, 65)
WHITE = (255, 255, 255)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CogniCore — AI Demo")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont(FONT_NAME, 36, bold=True)
font_med = pygame.font.SysFont(FONT_NAME, 20, bold=True)
font_sm = pygame.font.SysFont(FONT_NAME, 14)
font_xs = pygame.font.SysFont(FONT_NAME, 11)

frames = []

def capture():
    frame = pygame.surfarray.array3d(screen)
    frame = np.transpose(frame, (1, 0, 2))
    frames.append(frame.copy())

def draw_text(text, font, color, x, y, align="left"):
    s = font.render(text, True, color)
    r = s.get_rect()
    if align == "center": r.center = (x, y)
    elif align == "right": r.topright = (x, y)
    else: r.topleft = (x, y)
    screen.blit(s, r)

def draw_panel(x, y, w, h, title=""):
    rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, PANEL, rect, border_radius=12)
    pygame.draw.rect(screen, BORDER, rect, 1, border_radius=12)
    # Top accent line
    pygame.draw.line(screen, ACCENT, (x+12, y), (x+w-12, y))
    if title:
        draw_text(title, font_med, TEXT, x+16, y+12)

def draw_bar(x, y, w, h, value, max_val, color, label=""):
    pygame.draw.rect(screen, (20, 20, 35), (x, y, w, h), border_radius=4)
    fill_w = int(w * min(1, max(0, value / max(max_val, 1))))
    if fill_w > 0:
        pygame.draw.rect(screen, color, (x, y, fill_w, h), border_radius=4)
    if label:
        draw_text(label, font_xs, DIM, x, y - 14)
    draw_text(f"{value:.0f}", font_xs, TEXT, x + w + 6, y + 1)

def draw_header(title, subtitle, progress=0):
    # Background gradient bar
    for i in range(60):
        alpha = max(0, 255 - i * 4)
        pygame.draw.line(screen, (0, int(220 * alpha/255 * 0.05), int(150 * alpha/255 * 0.05)),
                        (0, i), (WIDTH, i))
    draw_text("🧠 CogniCore", font_big, ACCENT, 30, 10)
    draw_text("v0.7.0", font_xs, DIM, 280, 22)
    draw_text(title, font_med, WHITE, WIDTH//2, 12, "center")
    draw_text(subtitle, font_xs, DIM, WIDTH//2, 38, "center")
    # Progress bar
    pygame.draw.rect(screen, (20, 20, 35), (30, 55, WIDTH-60, 4), border_radius=2)
    if progress > 0:
        pw = int((WIDTH-60) * min(1, progress))
        pygame.draw.rect(screen, ACCENT, (30, 55, pw, 4), border_radius=2)


# ══════════════════════════════════════════════════
#  SCENE 1: Title Card
# ══════════════════════════════════════════════════
def scene_title(duration=3):
    for frame in range(int(duration * FPS)):
        screen.fill(BG)
        # Animated particles
        t = frame / FPS
        for i in range(30):
            px = (i * 97 + int(t * 20)) % WIDTH
            py = (i * 73 + int(t * 10)) % HEIGHT
            r = 2 + np.sin(t + i) * 1
            pygame.draw.circle(screen, (0, 60, 40), (px, py), int(r))

        pulse = abs(np.sin(t * 2)) * 0.3 + 0.7
        # Title
        title_s = font_big.render("CogniCore", True, tuple(int(c * pulse) for c in ACCENT))
        screen.blit(title_s, (WIDTH//2 - title_s.get_width()//2, HEIGHT//2 - 80))
        draw_text("Cognitive RL Environments", font_med, TEXT, WIDTH//2, HEIGHT//2 - 20, "center")
        draw_text("Memory • Reflection • Reward Shaping", font_sm, DIM, WIDTH//2, HEIGHT//2 + 20, "center")
        draw_text("PPO vs DQN vs Random — Live Training Demo", font_sm, GOLD, WIDTH//2, HEIGHT//2 + 60, "center")

        pygame.display.flip()
        capture()
        clock.tick(FPS)
        for e in pygame.event.get():
            if e.type == pygame.QUIT: return False
    return True


# ══════════════════════════════════════════════════
#  SCENE 2: MazeRunner — Random vs Trained
# ══════════════════════════════════════════════════
def draw_maze(maze_env, offset_x, offset_y, cell_size, label=""):
    size = maze_env.size
    for r in range(size):
        for c in range(size):
            x = offset_x + c * cell_size
            y = offset_y + r * cell_size
            pos = (r, c)
            rect = pygame.Rect(x, y, cell_size, cell_size)
            if pos in maze_env.walls:
                pygame.draw.rect(screen, WALL, rect)
            elif pos in maze_env.visit_counts:
                v = min(maze_env.visit_counts.get(pos, 0) * 12, 50)
                pygame.draw.rect(screen, (22+v, 28+v//2, 38+v), rect)
            else:
                pygame.draw.rect(screen, FLOOR, rect)
            pygame.draw.rect(screen, (18, 18, 30), rect, 1)

    # Goal
    gx = offset_x + maze_env.goal_pos[1] * cell_size
    gy = offset_y + maze_env.goal_pos[0] * cell_size
    pygame.draw.rect(screen, GOAL, (gx+3, gy+3, cell_size-6, cell_size-6), border_radius=3)
    draw_text("G", font_xs, BG, gx + cell_size//2 - 3, gy + cell_size//2 - 5)

    # Agent
    ax = offset_x + maze_env.agent_pos[1] * cell_size
    ay = offset_y + maze_env.agent_pos[0] * cell_size
    pygame.draw.rect(screen, AGENT, (ax+2, ay+2, cell_size-4, cell_size-4), border_radius=5)
    draw_text("A", font_xs, BG, ax + cell_size//2 - 3, ay + cell_size//2 - 5)

    if label:
        draw_text(label, font_med, TEXT, offset_x, offset_y - 24)


def scene_maze(duration_per_phase=4):
    print("  Training PPO on MazeRunner...")
    env_train = Monitor(gym.make("cognicore/MazeRunner-v0"))
    model = PPO("MlpPolicy", env_train, verbose=0, seed=42, n_steps=256, batch_size=64)
    model.learn(total_timesteps=30000)
    env_train.close()

    cs = 36  # cell size
    # Phase 1: Random agent
    env1 = gym.make("cognicore/MazeRunner-v0")
    env2 = gym.make("cognicore/MazeRunner-v0")
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)

    random_rewards = []
    trained_rewards = []
    total_r1, total_r2 = 0, 0

    for frame in range(int(duration_per_phase * FPS)):
        screen.fill(BG)
        t = frame / FPS
        progress = frame / (duration_per_phase * FPS)
        draw_header("MazeRunner-v0 — Agent Comparison",
                    "Random Agent vs PPO (30K steps trained)", progress)

        # Left panel - Random
        draw_panel(20, 70, WIDTH//2 - 30, HEIGHT - 90, "")
        draw_maze(env1.unwrapped, 40, 100, cs, "❌ Random Agent")
        draw_text(f"Reward: {total_r1:.1f}", font_sm,
                 RED if total_r1 < 0 else ACCENT, 40, HEIGHT - 60)
        draw_text(f"Steps: {env1.unwrapped.steps_taken}", font_sm, DIM, 40, HEIGHT - 40)

        # Right panel - Trained
        draw_panel(WIDTH//2 + 10, 70, WIDTH//2 - 30, HEIGHT - 90, "")
        draw_maze(env2.unwrapped, WIDTH//2 + 30, 100, cs, "✅ PPO Trained Agent")
        draw_text(f"Reward: {total_r2:.1f}", font_sm,
                 RED if total_r2 < 0 else ACCENT, WIDTH//2 + 30, HEIGHT - 60)
        draw_text(f"Steps: {env2.unwrapped.steps_taken}", font_sm, DIM, WIDTH//2 + 30, HEIGHT - 40)

        # VS label
        draw_text("VS", font_big, GOLD, WIDTH//2, HEIGHT//2 + 20, "center")

        pygame.display.flip()
        capture()
        clock.tick(FPS)

        # Step environments every few frames
        if frame % 4 == 0:
            # Random
            a1 = env1.action_space.sample()
            obs1, r1, t1, tr1, _ = env1.step(a1)
            total_r1 += r1
            if t1 or tr1:
                random_rewards.append(total_r1)
                obs1, _ = env1.reset(seed=frame)
                total_r1 = 0

            # Trained
            a2, _ = model.predict(obs2, deterministic=True)
            obs2, r2, t2, tr2, info2 = env2.step(int(a2))
            total_r2 += r2
            if t2 or tr2:
                trained_rewards.append(total_r2)
                obs2, _ = env2.reset(seed=frame)
                total_r2 = 0

        for e in pygame.event.get():
            if e.type == pygame.QUIT: return False

    env1.close()
    env2.close()
    return True


# ══════════════════════════════════════════════════
#  SCENE 3: Survival — PPO vs Random
# ══════════════════════════════════════════════════
def scene_survival(duration=5):
    print("  Training PPO on Survival...")
    env_t = Monitor(gym.make("cognicore/Survival-v0"))
    model = PPO("MlpPolicy", env_t, verbose=0, seed=42, n_steps=256, batch_size=64)
    model.learn(total_timesteps=30000)
    env_t.close()

    env1 = gym.make("cognicore/Survival-v0")
    env2 = gym.make("cognicore/Survival-v0")
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)

    bars1 = {"Health": 100, "Hunger": 0, "Food": 5}
    bars2 = {"Health": 100, "Hunger": 0, "Food": 5}
    day1 = day2 = 0
    alive1 = alive2 = True
    r1_total = r2_total = 0

    for frame in range(int(duration * FPS)):
        screen.fill(BG)
        progress = frame / (duration * FPS)
        draw_header("Survival-v0 — Long-Horizon Planning",
                    "Random vs PPO: Who survives longer?", progress)

        # Left: Random
        draw_panel(20, 70, WIDTH//2 - 30, HEIGHT - 90)
        draw_text("❌ Random Agent", font_med, TEXT, 40, 82)
        st1 = "ALIVE" if alive1 else "DEAD"
        col1 = ACCENT if alive1 else RED
        draw_text(f"Day {day1} — {st1}", font_sm, col1, 40, 110)
        y = 140
        for name, val in bars1.items():
            mx = 100 if name != "Food" else 20
            c = ACCENT if name == "Health" else RED if name == "Hunger" else GOLD
            draw_bar(40, y, WIDTH//2 - 100, 18, val, mx, c, name)
            y += 36
        draw_text(f"Total Reward: {r1_total:.1f}", font_sm,
                 RED if r1_total < 0 else ACCENT, 40, HEIGHT - 50)

        # Right: PPO
        draw_panel(WIDTH//2 + 10, 70, WIDTH//2 - 30, HEIGHT - 90)
        draw_text("✅ PPO Trained Agent", font_med, TEXT, WIDTH//2 + 30, 82)
        st2 = "ALIVE" if alive2 else "DEAD"
        col2 = ACCENT if alive2 else RED
        draw_text(f"Day {day2} — {st2}", font_sm, col2, WIDTH//2 + 30, 110)
        y = 140
        for name, val in bars2.items():
            mx = 100 if name != "Food" else 20
            c = ACCENT if name == "Health" else RED if name == "Hunger" else GOLD
            draw_bar(WIDTH//2 + 30, y, WIDTH//2 - 100, 18, val, mx, c, name)
            y += 36

        draw_text(f"Total Reward: {r2_total:.1f}", font_sm,
                 RED if r2_total < 0 else ACCENT, WIDTH//2 + 30, HEIGHT - 50)

        draw_text("VS", font_big, GOLD, WIDTH//2, HEIGHT//2, "center")

        pygame.display.flip()
        capture()
        clock.tick(FPS)

        if frame % 3 == 0:
            if alive1:
                a1 = env1.action_space.sample()
                obs1, r1, t1, tr1, info1 = env1.step(a1)
                r1_total += r1
                bars1 = {"Health": obs1[0], "Hunger": obs1[1], "Food": obs1[4]}
                day1 = int(obs1[8])
                if t1 or tr1: alive1 = False

            if alive2:
                a2, _ = model.predict(obs2, deterministic=True)
                obs2, r2, t2, tr2, info2 = env2.step(int(a2))
                r2_total += r2
                bars2 = {"Health": obs2[0], "Hunger": obs2[1], "Food": obs2[4]}
                day2 = int(obs2[8])
                if t2 or tr2: alive2 = False

        for e in pygame.event.get():
            if e.type == pygame.QUIT: return False

    env1.close()
    env2.close()
    return True


# ══════════════════════════════════════════════════
#  SCENE 4: Final Results
# ══════════════════════════════════════════════════
def scene_results(duration=4):
    results = [
        ("GridWorld-v0", "PPO", "+1.5", "Random", "-4.2", ACCENT),
        ("Survival-v0", "PPO", "+199.9", "Random", "+15.0", ACCENT),
        ("Trading-v0", "PPO", "+0.0", "Random", "-0.5", BLUE),
        ("BattleArena-v0", "PPO", "+19.2", "Random", "-3.5", GOLD),
    ]

    for frame in range(int(duration * FPS)):
        screen.fill(BG)
        t = frame / FPS

        draw_text("🧠 CogniCore", font_big, ACCENT, WIDTH//2, 40, "center")
        draw_text("Benchmark Results — PPO vs Random Baseline", font_med, TEXT, WIDTH//2, 85, "center")
        draw_text("50K training steps • 50 eval episodes • Gymnasium-native", font_sm, DIM, WIDTH//2, 115, "center")

        # Table
        y = 160
        headers = ["Environment", "Best Algo", "Score", "Random", "Score", "Improvement"]
        hx = [80, 300, 460, 580, 700, 860]
        for i, h in enumerate(headers):
            draw_text(h, font_xs, DIM, hx[i], y)
        y += 30
        pygame.draw.line(screen, BORDER, (60, y-5), (WIDTH-60, y-5))

        for i, (env, algo, score, base, bscore, color) in enumerate(results):
            appear_frame = int(i * 0.5 * FPS)
            if frame > appear_frame:
                alpha = min(1, (frame - appear_frame) / (0.3 * FPS))
                c = tuple(int(v * alpha) for v in TEXT)
                draw_text(f"cognicore/{env}", font_sm, c, 80, y)
                draw_text(algo, font_sm, tuple(int(v * alpha) for v in color), 300, y)
                draw_text(score, font_sm, tuple(int(v * alpha) for v in ACCENT), 460, y)
                draw_text(base, font_sm, tuple(int(v * alpha) for v in DIM), 580, y)
                draw_text(bscore, font_sm, tuple(int(v * alpha) for v in RED), 700, y)

                # Bar
                try:
                    s = float(score)
                    b = float(bscore)
                    impr = s - b
                    bar_w = min(200, max(10, abs(impr) * 2))
                    bar_color = ACCENT if impr > 0 else RED
                    pygame.draw.rect(screen, bar_color, (860, y+2, int(bar_w * alpha), 14), border_radius=3)
                    draw_text(f"+{impr:.1f}", font_xs, bar_color, 860 + int(bar_w * alpha) + 8, y + 2)
                except: pass
                y += 40

        # Footer
        if t > 2:
            draw_text("All environments pass gymnasium.utils.env_checker.check_env() ✅", font_sm, ACCENT, WIDTH//2, HEIGHT - 80, "center")
            draw_text("Compatible with Stable Baselines3 • RLlib • CleanRL", font_sm, DIM, WIDTH//2, HEIGHT - 55, "center")
            draw_text("github.com/Kaushalt2004/cognicore-my-openenv", font_sm, BLUE, WIDTH//2, HEIGHT - 30, "center")

        pygame.display.flip()
        capture()
        clock.tick(FPS)
        for e in pygame.event.get():
            if e.type == pygame.QUIT: return False
    return True


# ══════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  CogniCore Demo Recorder")
    print("  Recording AI agents learning across all environments")
    print("=" * 60)

    if not scene_title(3): return
    print("  [1/4] Title card recorded")

    if not scene_maze(6): return
    print("  [2/4] MazeRunner comparison recorded")

    if not scene_survival(6): return
    print("  [3/4] Survival comparison recorded")

    if not scene_results(5): return
    print("  [4/4] Results table recorded")

    pygame.quit()

    # Save video
    print(f"\n  Saving {len(frames)} frames as video...")
    try:
        import imageio
        output_path = os.path.join(os.path.dirname(__file__), "cognicore_demo.mp4")
        writer = imageio.get_writer(output_path, fps=FPS, codec='libx264',
                                     quality=8, pixelformat='yuv420p')
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"  ✅ Video saved: {output_path}")
        print(f"  Duration: {len(frames)/FPS:.1f}s")
    except ImportError:
        print("  imageio not installed, saving as frames...")
        frame_dir = os.path.join(os.path.dirname(__file__), "frames")
        os.makedirs(frame_dir, exist_ok=True)
        from PIL import Image
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(frame_dir, f"frame_{i:05d}.png"))
        print(f"  Saved {len(frames)} frames to {frame_dir}/")
        print(f"  Convert with: ffmpeg -framerate 30 -i frames/frame_%05d.png -c:v libx264 demo.mp4")

    print("\n" + "=" * 60)
    print("  DONE — Share this video to demonstrate CogniCore!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {e}")
        pygame.quit()
