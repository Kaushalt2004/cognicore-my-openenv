"""
FastAPI server for CogniCore AI Safety Monitor.

Serves both the interactive dashboard UI and the OpenEnv API endpoints.

Endpoints:
  GET  /        — Interactive Dashboard UI
  GET  /health  — Health check
  POST /reset   — Reset environment for a new episode
  POST /step    — Take one step (submit a classification)
  GET  /state   — Get current environment state
  GET  /api     — API info (JSON)
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any

from env.environment import AISafetyEnv


# ─── App setup ────────────────────────────────────────────────

app = FastAPI(
    title="CogniCore AI Safety Monitor",
    description="An OpenEnv environment where agents learn to detect unsafe AI responses.",
    version="1.0.0",
)

# Global environment instance
env = AISafetyEnv()

# Path to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ─── Request / Response models ──────────────────────────────

class ResetRequest(BaseModel):
    task: str = "binary_safety_classification"
    difficulty: Optional[str] = None


class StepRequest(BaseModel):
    classification: str
    confidence: float = 0.5
    reasoning: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "healthy"
    environment: str = "cognicore-ai-safety-monitor"
    version: str = "1.0.0"


# ─── Dashboard UI ──────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the interactive dashboard."""
    dashboard_path = os.path.join(BASE_DIR, "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    # Fallback if dashboard.html not found
    return HTMLResponse(content="""
    <html>
    <head><title>CogniCore AI Safety Monitor</title></head>
    <body style="font-family:system-ui;background:#0a0e17;color:#e8ecf4;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;">
        <div style="text-align:center;">
            <h1>🧠 CogniCore AI Safety Monitor</h1>
            <p>Environment is running. Use the API endpoints:</p>
            <p><code>/health</code> · <code>/reset</code> · <code>/step</code> · <code>/state</code> · <code>/docs</code></p>
        </div>
    </body>
    </html>
    """)


# ─── API Endpoints ──────────────────────────────────────────

@app.get("/api")
def api_info() -> Dict[str, Any]:
    """API info endpoint — environment metadata."""
    return {
        "name": "cognicore-ai-safety-monitor",
        "version": "1.0.0",
        "description": "CogniCore AI Safety Monitor — classify AI responses as SAFE, UNSAFE, or NEEDS_REVIEW",
        "endpoints": {
            "health": "/health",
            "reset": "/reset",
            "step": "/step",
            "state": "/state",
            "docs": "/docs",
        },
        "tasks": [
            "binary_safety_classification",
            "nuanced_safety_detection",
            "adversarial_safety_monitoring",
        ],
    }


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check — required by OpenEnv."""
    return HealthResponse()


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    """Reset the environment for a new episode."""
    try:
        observation = env.reset(task=request.task, difficulty=request.difficulty)
        return {
            "observation": observation,
            "done": False,
            "reward": 0.0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    """Take one step — submit a safety classification."""
    try:
        action = {
            "classification": request.classification,
            "confidence": request.confidence,
            "reasoning": request.reasoning,
        }
        observation, reward, done, info = env.step(action)
        return {
            "observation": observation,
            "reward": round(reward, 4),
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state() -> Dict[str, Any]:
    """Get the current environment state."""
    return env.state()


# ─── Run with uvicorn ───────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
