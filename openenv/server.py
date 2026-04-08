"""
FastAPI server exposing the PRReviewEnvironment over HTTP.

Endpoints:
  POST /reset          → Observation (starts/resets episode)
  POST /step           → {observation, reward, done, info}
  GET  /state          → EpisodeState
  GET  /health         → {"status": "ok"}
  GET  /               → basic info

HuggingFace Spaces compatibility: listens on 0.0.0.0:7860
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.environment.env import PRReviewEnvironment
from app.models.schemas import Action, EpisodeState, Observation, Reward

app = FastAPI(
    title="GitHub PR Code Review Environment",
    description="OpenEnv environment simulating GitHub pull request code review",
    version="1.0.0",
    tags_metadata=[{"name": "openenv"}],
)

# Allow cross-origin requests (required for HF Spaces iframe usage)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment instance per server process (stateful)
_envs: dict[str, PRReviewEnvironment] = {}


def _get_env(task_id: str) -> PRReviewEnvironment:
    """Return (or create) the environment for the given task_id."""
    if task_id not in _envs:
        _envs[task_id] = PRReviewEnvironment(task_id=task_id)
    return _envs[task_id]


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task1"


class StepRequest(BaseModel):
    task_id: str = "task1"
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "GitHub PR Code Review Environment",
        "version": "1.0.0",
        "tasks": "task1 (easy) | task2 (medium) | task3 (hard)",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest = ResetRequest()) -> Observation:
    """Reset the environment and return the initial observation (bare Observation per OpenEnv spec)."""
    env = _get_env(req.task_id)
    return env.reset()


@app.post("/step")
def step(req: StepRequest):
    env = _get_env(req.task_id)
    obs, reward, done, info = env.step(req.action)

    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state", response_model=EpisodeState)
def state(task_id: str = "task1") -> EpisodeState:
    """Return the current internal episode state."""
    env = _get_env(task_id)
    return env.state()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
