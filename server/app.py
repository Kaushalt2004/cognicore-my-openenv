"""
FastAPI application for CogniCore AI Safety Monitor.

Uses OpenEnv SDK types (Action, Observation, State) for spec compliance
but manages session state server-side for HTTP-based multi-step episodes.

Endpoints:
  GET  /        — Interactive Dashboard UI
  GET  /health  — Health check (OpenEnv standard)
  POST /reset   — Reset environment for a new episode
  POST /step    — Take one step (submit a classification)
  GET  /state   — Get current environment state
  GET  /api     — API info (JSON)
  GET  /docs    — FastAPI auto-docs
"""

import os
import sys
import logging
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any

from models import SafetyAction, SafetyObservation, SafetyState
from server.environment import SafetyMonitorEnvironment

_logger = logging.getLogger(__name__)


# ─── App setup ────────────────────────────────────────────────

app = FastAPI(
    title="CogniCore AI Safety Monitor",
    description="An OpenEnv environment where agents learn to detect unsafe AI responses. "
                "Uses memory-augmented context, metacognitive reflection, and multi-dimensional grading.",
    version="2.0.0",
)

# Global environment instance (persists across requests)
env = SafetyMonitorEnvironment()


# ─── Request / Response models ──────────────────────────────

class ResetRequest(BaseModel):
    task: str = "binary_safety_classification"
    difficulty: Optional[str] = None

class StepRequest(BaseModel):
    classification: str
    confidence: float = 0.5
    severity: str = "medium"
    reasoning: Optional[str] = None
    manipulation_type: Optional[str] = None

class HealthResponse(BaseModel):
    status: str = "healthy"
    environment: str = "cognicore-ai-safety-monitor"
    version: str = "2.0.0"


# ─── Dashboard UI ──────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the interactive dashboard."""
    dashboard_path = PROJECT_ROOT / "dashboard.html"
    if dashboard_path.exists():
        return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="""
    <html>
    <head><title>CogniCore AI Safety Monitor</title></head>
    <body style="font-family:system-ui;background:#0a0e17;color:#e8ecf4;
                 display:flex;align-items:center;justify-content:center;
                 height:100vh;margin:0;">
        <div style="text-align:center;">
            <h1>🧠 CogniCore AI Safety Monitor v2.0</h1>
            <p>OpenEnv environment is running.</p>
            <p><code>/health</code> · <code>/reset</code> · <code>/step</code> · 
               <code>/state</code> · <code>/docs</code></p>
        </div>
    </body>
    </html>
    """)


# ─── API Endpoints ──────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check — required by OpenEnv."""
    return HealthResponse()


@app.get("/api")
def api_info() -> Dict[str, Any]:
    """Environment metadata."""
    return {
        "name": "cognicore-ai-safety-monitor",
        "version": "2.0.0",
        "description": "CogniCore AI Safety Monitor — classify AI responses as SAFE, UNSAFE, or NEEDS_REVIEW",
        "tasks": [
            "binary_safety_classification",
            "nuanced_safety_detection",
            "adversarial_safety_monitoring",
        ],
        "actions": ["SAFE", "UNSAFE", "NEEDS_REVIEW"],
        "features": [
            "memory_augmented", "reflection_engine", "streak_penalty",
            "confidence_calibration", "severity_scoring", "manipulation_detection",
        ],
        "scoring": {
            "easy": "binary + confidence calibration",
            "medium": "accuracy(60%) + severity(20%) + calibration(20%)",
            "hard": "accuracy(40%) + manipulation_id(25%) + reasoning(15%) + calibration(20%)",
        },
    }


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    """Reset the environment for a new episode."""
    try:
        obs = env.reset(task=request.task, difficulty=request.difficulty)
        return {
            "observation": obs.model_dump() if hasattr(obs, 'model_dump') else _obs_to_dict(obs),
            "reward": None,
            "done": False,
            "info": {},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    """Take one step — submit a safety classification.
    
    Returns structured reward with full penalty breakdown and step info.
    """
    try:
        action = SafetyAction(
            classification=request.classification,
            confidence=request.confidence,
            severity=request.severity,
            reasoning=request.reasoning,
        )
        # manipulation_type passed separately (not part of OpenEnv Action schema)
        obs = env.step(action, manipulation_type=request.manipulation_type)
        
        # Get structured reward and step info
        reward_data = env.last_reward
        step_info = env.last_step_info
        
        return {
            "observation": obs.model_dump() if hasattr(obs, 'model_dump') else _obs_to_dict(obs),
            "reward": reward_data.model_dump() if reward_data else {"value": obs.reward},
            "done": obs.done,
            "info": step_info.model_dump() if step_info else {},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state() -> Dict[str, Any]:
    """Get the current environment state."""
    s = env.state
    return s.model_dump() if hasattr(s, 'model_dump') else vars(s)


def _obs_to_dict(obs) -> dict:
    """Fallback observation serializer."""
    return {k: v for k, v in vars(obs).items() if not k.startswith('_')}


# ─── Run with uvicorn ───────────────────────────────────────

def main():
    """Entry point for running via uv run or python -m."""
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
