"""CogniCore Live Demo — Real cognitive environment testing via web UI."""
import asyncio, json, os
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
import cognicore

app = FastAPI(title="CogniCore Live Demo")

@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "cognitive_demo.html"))

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = json.loads(await ws.receive_text())
            if data["action"] == "run_cognitive_demo":
                await run_cognitive_demo(ws)
            elif data["action"] == "run_single":
                await run_single_env(ws, data.get("env", "CodeDebugging-v1"), data.get("episodes", 10))
    except Exception as e:
        print(f"WS error: {e}")

async def send(ws, d):
    try: await ws.send_text(json.dumps(d))
    except: pass

def R(r):
    return float(getattr(r, "total", r))

async def run_single_env(ws, env_name, episodes):
    await send(ws, {"type": "status", "msg": f"Running {env_name}..."})
    env = cognicore.make(env_name)
    
    for ep in range(episodes):
        obs = env.reset()
        mem_count = len(obs.get("memory_context", []))
        
        # Smart heuristic based on env type
        if "CodeDebugging" in env_name:
            lines = obs["buggy_code"].strip().split("\n")
            action = {"bug_line": len(lines)//2, "fix_type": obs.get("category", "syntax_error")}
            task_text = obs["buggy_code"][:150]
        elif "Safety" in env_name:
            txt = str(obs).lower()
            bad = ["hack","attack","kill","bomb","weapon","steal","exploit","harm"]
            guess = "UNSAFE" if any(w in txt for w in bad) else "SAFE"
            action = {"classification": guess}
            task_text = str(list(obs.values())[0])[:150]
        elif "Math" in env_name:
            action = {"answer": "42"}
            task_text = str(list(obs.values())[0])[:150]
        elif "Conversation" in env_name:
            action = {"response": "I understand your concern and want to help."}
            task_text = str(list(obs.values())[0])[:150]
        else:
            action = "default"
            task_text = str(obs)[:150]
        
        _, reward, _, _, info = env.step(action)
        er = info.get("eval_result", {})
        rc = info.get("reward_components", {})
        
        await send(ws, {
            "type": "episode",
            "env": env_name,
            "episode": ep + 1,
            "task": task_text,
            "correct": er.get("correct", False),
            "reward": R(reward),
            "ground_truth": er.get("ground_truth", ""),
            "predicted": er.get("predicted", ""),
            "memory_entries": mem_count,
            "memory_bonus": rc.get("memory_bonus", 0),
            "reflection_bonus": rc.get("reflection_bonus", 0),
            "category": er.get("category", obs.get("category", "")),
        })
        await asyncio.sleep(0.05)
    
    stats = env.episode_stats()
    await send(ws, {
        "type": "env_done",
        "env": env_name,
        "stats": {
            "episodes": stats.episode_number,
            "accuracy": stats.accuracy,
            "correct": stats.correct_count,
            "memory_entries": stats.memory_entries_created,
            "reflections": stats.reflection_hints_given,
        }
    })

async def run_cognitive_demo(ws):
    envs = [
        ("CodeDebugging-v1", 10),
        ("SafetyClassification-v1", 10),
        ("MathReasoning-Easy-v1", 5),
        ("Conversation-Easy-v1", 5),
    ]
    for env_name, eps in envs:
        await run_single_env(ws, env_name, eps)
        await asyncio.sleep(0.1)
    
    await send(ws, {"type": "demo_done"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500)
