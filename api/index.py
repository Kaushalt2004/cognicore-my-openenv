"""
CogniCore API — Vercel Serverless Function

This wraps the CogniCore FastAPI server for Vercel deployment.
"""
import sys
import os

# Add parent directory to path so cognicore can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import uuid
from collections import defaultdict

app = FastAPI(title="CogniCore API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage
sessions = {}

try:
    import cognicore
    COGNICORE_AVAILABLE = True
except ImportError:
    COGNICORE_AVAILABLE = False


@app.get("/api")
async def root():
    return {
        "name": "CogniCore API",
        "version": "0.1.0",
        "status": "running",
        "cognicore_available": COGNICORE_AVAILABLE,
        "exports": 71,
        "environments": 24,
        "cli_commands": 22,
    }


@app.get("/api/health")
async def health():
    return {"status": "healthy", "cognicore": COGNICORE_AVAILABLE}


@app.get("/api/envs")
async def list_envs():
    if not COGNICORE_AVAILABLE:
        return {"environments": []}
    envs = cognicore.list_envs()
    return {
        "environments": [
            {"id": e, "description": _get_env_desc(e)}
            for e in envs
        ]
    }


@app.post("/api/envs/{env_id}/create")
async def create_session(env_id: str, request: Request):
    if not COGNICORE_AVAILABLE:
        return JSONResponse({"error": "cognicore not available"}, 500)

    body = await request.json() if await request.body() else {}
    difficulty = body.get("difficulty", "easy")

    try:
        env = cognicore.make(env_id, difficulty=difficulty)
        sid = str(uuid.uuid4())[:8]
        sessions[sid] = {"env": env, "env_id": env_id, "difficulty": difficulty}
        return {"session_id": sid, "env_id": env_id, "difficulty": difficulty}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 400)


@app.post("/api/sessions/{sid}/reset")
async def reset_session(sid: str):
    if sid not in sessions:
        return JSONResponse({"error": "session not found"}, 404)
    env = sessions[sid]["env"]
    obs = env.reset()
    return {"observation": _serialize_obs(obs)}


@app.post("/api/sessions/{sid}/step")
async def step_session(sid: str, request: Request):
    if sid not in sessions:
        return JSONResponse({"error": "session not found"}, 404)

    body = await request.json()
    action = body.get("action", {})
    env = sessions[sid]["env"]

    obs, reward, done, truncated, info = env.step(action)

    return {
        "observation": _serialize_obs(obs),
        "reward": {
            "total": reward.total,
            "base_score": reward.base_score,
            "memory_bonus": reward.memory_bonus,
            "reflection_bonus": reward.reflection_bonus,
            "streak_penalty": reward.streak_penalty,
            "propose_bonus": reward.propose_bonus,
            "novelty_bonus": reward.novelty_bonus,
            "confidence_calibration": reward.confidence_calibration,
            "time_decay": reward.time_decay,
        },
        "done": done,
        "truncated": truncated,
        "info": _serialize_info(info),
    }


@app.get("/api/sessions/{sid}/stats")
async def session_stats(sid: str):
    if sid not in sessions:
        return JSONResponse({"error": "session not found"}, 404)
    env = sessions[sid]["env"]
    stats = env.episode_stats()
    return {
        "score": env.get_score(),
        "accuracy": stats.accuracy,
        "steps": stats.steps,
        "correct_count": stats.correct_count,
        "memory_entries_created": stats.memory_entries_created,
    }


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str):
    if sid in sessions:
        del sessions[sid]
    return {"deleted": True}


@app.get("/api/stats")
async def platform_stats():
    return {
        "version": "0.1.0",
        "exports": 71,
        "environments": 24,
        "cli_commands": 22,
        "tests": 277,
        "doctor_checks": 55,
        "active_sessions": len(sessions),
    }


def _get_env_desc(env_id):
    try:
        env = cognicore.make(env_id)
        return getattr(env, "description", env_id)
    except:
        return env_id


def _serialize_obs(obs):
    if isinstance(obs, dict):
        return {k: str(v) if not isinstance(v, (str, int, float, bool, list)) else v for k, v in obs.items()}
    return str(obs)


def _serialize_info(info):
    result = {}
    for k, v in info.items():
        if hasattr(v, "__dict__"):
            result[k] = {ak: av for ak, av in v.__dict__.items() if isinstance(av, (str, int, float, bool, type(None)))}
        elif isinstance(v, (str, int, float, bool, type(None))):
            result[k] = v
    return result
