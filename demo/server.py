"""CogniCore Demo Server — Streams real environment state for visual rendering."""
import asyncio, json, os, time, numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
import cognicore.gym
import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

app = FastAPI()

@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "demo.html"))

class LiveCallback(BaseCallback):
    """Collect per-episode stats during training."""
    def __init__(self):
        super().__init__()
        self.episodes = []
        self._ep_reward = 0
        self._ep_len = 0
    def _on_step(self):
        for i, done in enumerate(self.locals.get("dones", [])):
            self._ep_reward += self.locals["rewards"][i]
            self._ep_len += 1
            if done:
                self.episodes.append({"r": round(float(self._ep_reward), 2), "l": self._ep_len, "t": self.num_timesteps})
                self._ep_reward = 0; self._ep_len = 0
        return True

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg["action"] == "train_and_visualize":
                await train_and_visualize(ws, msg)
    except Exception as e:
        print(f"WS: {e}")

async def send(ws, d):
    try: await ws.send_text(json.dumps(d))
    except: pass

async def train_and_visualize(ws, msg):
    env_id = msg.get("env", "cognicore/GridWorld-v0")
    steps = msg.get("steps", 25000)
    algo_name = msg.get("algo", "PPO")
    algos = {"PPO": PPO, "DQN": DQN, "A2C": A2C}
    cls = algos.get(algo_name, PPO)

    await send(ws, {"type": "status", "phase": "training", "msg": f"Training {algo_name} on {env_id}..."})

    # Phase 1: Train
    env_train = Monitor(gym.make(env_id))
    cb = LiveCallback()
    t0 = time.time()
    model = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _train(cls, env_train, steps, cb))
    dt = time.time() - t0
    env_train.close()

    # Stream training curve
    for i, ep in enumerate(cb.episodes):
        await send(ws, {"type": "train_ep", "i": i, "reward": ep["r"], "length": ep["l"], "timestep": ep["t"]})
        if i % 5 == 0:
            await asyncio.sleep(0.01)
    
    await send(ws, {"type": "status", "phase": "trained", "msg": f"{algo_name} trained in {dt:.1f}s ({len(cb.episodes)} episodes)"})

    # Phase 2: Visualize trained agent
    await send(ws, {"type": "status", "phase": "visualizing", "msg": "Running trained agent..."})
    env_vis = gym.make(env_id)
    obs, _ = env_vis.reset(seed=42)
    uw = env_vis.unwrapped
    total_r = 0

    # Send initial grid state
    await send_env_state(ws, uw, env_id, 0, total_r, "start")

    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, te, tr, info = env_vis.step(int(action))
        total_r += float(r)
        
        await send_env_state(ws, uw, env_id, step + 1, total_r, "step")
        await asyncio.sleep(0.06)  # ~15fps visual

        if te or tr:
            await send(ws, {"type": "episode_end", "steps": step + 1, "reward": round(total_r, 2), "success": te and total_r > 0})
            obs, _ = env_vis.reset()
            uw = env_vis.unwrapped
            total_r = 0
            await send_env_state(ws, uw, env_id, 0, 0, "reset")
            await asyncio.sleep(0.3)

    env_vis.close()

    # Phase 3: Random baseline for comparison
    await send(ws, {"type": "status", "phase": "baseline", "msg": "Running random baseline..."})
    env_rand = gym.make(env_id)
    rand_rewards = []
    for _ in range(50):
        obs, _ = env_rand.reset()
        ep_r = 0
        for _ in range(200):
            obs, r, te, tr, _ = env_rand.step(env_rand.action_space.sample())
            ep_r += float(r)
            if te or tr: break
        rand_rewards.append(round(ep_r, 2))
    env_rand.close()

    # Trained eval
    env_eval = gym.make(env_id)
    trained_rewards = []
    for _ in range(50):
        obs, _ = env_eval.reset()
        ep_r = 0
        for _ in range(200):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, te, tr, _ = env_eval.step(int(a))
            ep_r += float(r)
            if te or tr: break
        trained_rewards.append(round(ep_r, 2))
    env_eval.close()

    await send(ws, {"type": "comparison", 
        "trained_mean": round(float(np.mean(trained_rewards)), 2),
        "trained_std": round(float(np.std(trained_rewards)), 2),
        "random_mean": round(float(np.mean(rand_rewards)), 2),
        "random_std": round(float(np.std(rand_rewards)), 2),
        "algo": algo_name, "env": env_id, "time": round(dt, 1),
        "trained_dist": trained_rewards[:20], "random_dist": rand_rewards[:20]
    })
    await send(ws, {"type": "done"})

def _train(cls, env, steps, cb):
    model = cls("MlpPolicy", env, verbose=0, seed=42)
    model.learn(total_timesteps=steps, callback=cb)
    return model

async def send_env_state(ws, uw, env_id, step, total_r, phase):
    """Extract and send the visual state of the environment."""
    state = {"type": "env_state", "step": step, "reward": round(total_r, 2), "phase": phase}
    
    if "GridWorld" in env_id:
        state["env_type"] = "grid"
        state["size"] = uw.size
        state["agent"] = list(uw.agent_pos)
        state["goal"] = list(uw.goal_pos)
        state["traps"] = [list(t) for t in uw.traps]
        state["visits"] = {f"{k[0]},{k[1]}": v for k, v in uw.visit_counts.items()}
    elif "MazeRunner" in env_id:
        state["env_type"] = "maze"
        state["size"] = uw.size
        state["agent"] = list(uw.agent_pos)
        state["goal"] = list(uw.goal_pos)
        state["walls"] = [[w[0], w[1]] for w in uw.walls]
        state["visits"] = {f"{k[0]},{k[1]}": v for k, v in uw.visit_counts.items()}
    elif "Survival" in env_id:
        state["env_type"] = "survival"
        state["agent"] = list(uw.agent_pos) if hasattr(uw, 'agent_pos') else [0, 0]
        state["health"] = float(uw.health) if hasattr(uw, 'health') else 100
        state["hunger"] = float(uw.hunger) if hasattr(uw, 'hunger') else 0
        state["day"] = int(uw.day) if hasattr(uw, 'day') else 0
    elif "Trading" in env_id:
        state["env_type"] = "trading"
        state["balance"] = float(uw.balance) if hasattr(uw, 'balance') else 1000
        state["position"] = int(uw.position) if hasattr(uw, 'position') else 0
    else:
        state["env_type"] = "generic"
    
    await send(ws, state)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500)
