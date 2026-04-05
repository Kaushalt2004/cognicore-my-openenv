"""
FastAPI server for CogniCore AI Safety Monitor.

Endpoints:
  GET  /health  — Health check
  POST /reset   — Reset environment for a new episode
  POST /step    — Take one step (submit a classification)
  GET  /state   — Get current environment state
"""

from fastapi import FastAPI, HTTPException
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


# ─── Endpoints ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check — required by OpenEnv."""
    return HealthResponse()


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    """Reset the environment for a new episode.

    Args:
        task: Task name (binary_safety_classification,
              nuanced_safety_detection, adversarial_safety_monitoring).
        difficulty: Override difficulty (easy, medium, hard).

    Returns:
        First observation.
    """
    try:
        observation = env.reset(task=request.task, difficulty=request.difficulty)
        return {"observation": observation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    """Take one step — submit a safety classification.

    Args:
        classification: SAFE, UNSAFE, or NEEDS_REVIEW.
        confidence: 0.0 to 1.0 confidence score.
        reasoning: Optional reasoning text.

    Returns:
        observation, reward, done, info.
    """
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
