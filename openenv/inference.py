"""
Baseline inference script for the GitHub PR Code Review Environment.

Environment variables (MANDATORY — do not rename):
  API_BASE_URL       Base URL for the OpenAI-compatible API
  MODEL_NAME         Model identifier
  HF_TOKEN           API key / HuggingFace token
  LOCAL_IMAGE_NAME   (optional) Docker image name for local env hosting

Logging format (STRICT — do not change):
  [START] task=<task_name> env=<env_name> model=<model_name>
  [STEP]  step=<int> action=<string> reward=<float> done=<bool> error=<string_or_null>
  [END]   success=<bool> steps=<int> score=<float> rewards=<list>

Runtime target: < 20 minutes on CPU (2 vCPU, 8 GB RAM)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Environment variables — EXACT names required by spec
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN") or "dummy"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # optional

# ---------------------------------------------------------------------------
# Validate required variables
# ---------------------------------------------------------------------------
if not API_BASE_URL:
    print("[WARN] API_BASE_URL not set, using default", file=sys.stderr)
if not MODEL_NAME:
    print("[WARN] MODEL_NAME not set, using default", file=sys.stderr)
if not HF_TOKEN:
    print("[WARN] HF_TOKEN not set, running in fallback mode", file=sys.stderr)

# ---------------------------------------------------------------------------
# OpenAI client — MANDATORY client configuration
# ---------------------------------------------------------------------------
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ---------------------------------------------------------------------------
# Environment setup
# Supports two modes:
#   1. Direct Python import (default — LOCAL_IMAGE_NAME blank)
#   2. Docker image via LOCAL_IMAGE_NAME (uses HTTP transport to local container)
# ---------------------------------------------------------------------------

ENV_NAME = "PRReviewEnvironment"
TASKS = ["task1", "task2", "task3"]

# Keep well below task max_steps so env always gets a chance to set final_score
# task1=8, task2=10, task3=12 — use 10 as safe cap (task3 gets up to 10 steps)
MAX_STEPS_PER_TASK = 10
REQUEST_TIMEOUT = 60   # seconds per LLM call
RETRY_WAIT = 5         # seconds to wait before retrying a failed LLM call
MAX_LLM_RETRIES = 2    # retry transient errors (rate limits, timeouts) this many times


def _build_env_client(task_id: str):
    """
    Return an environment interface for the given task.
    If LOCAL_IMAGE_NAME is set, connects to a locally running Docker container
    via HTTP; otherwise uses the Python environment directly.
    """
    if LOCAL_IMAGE_NAME:
        import requests  # type: ignore

        base = "http://localhost:7860"

        class DockerEnvClient:
            """Thin HTTP wrapper around the containerised environment."""

            def reset(self) -> dict[str, Any]:
                resp = requests.post(f"{base}/reset", json={"task_id": task_id}, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                # Handle both bare Observation and legacy wrapped {"observation": ...}
                return data.get("observation", data)

            def step(self, action: dict[str, Any]) -> tuple[dict, dict, bool, dict]:
                resp = requests.post(
                    f"{base}/step",
                    json={"task_id": task_id, "action": action},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["observation"], data["reward"], data["done"], data["info"]

            def state(self) -> dict[str, Any]:
                resp = requests.get(f"{base}/state", params={"task_id": task_id}, timeout=30)
                resp.raise_for_status()
                return resp.json()

        return DockerEnvClient()
    else:
        from app.environment.env import PRReviewEnvironment
        from app.models.schemas import Action  # noqa: F401

        class DirectEnvClient:
            """Wraps PRReviewEnvironment with a dict-based interface."""

            def __init__(self) -> None:
                self._env = PRReviewEnvironment(task_id=task_id)

            def reset(self) -> dict[str, Any]:
                return self._env.reset().model_dump()

            def step(self, action_dict: dict[str, Any]) -> tuple[dict, dict, bool, dict]:
                from app.models.schemas import Action
                action = Action(**action_dict)
                obs, reward, done, info = self._env.step(action)
                return obs.model_dump(), reward.model_dump(), done, info

            def state(self) -> dict[str, Any]:
                return self._env.state().model_dump()

        return DirectEnvClient()


# ---------------------------------------------------------------------------
# Observation normalisation
# ---------------------------------------------------------------------------

def _unwrap_obs(data: dict[str, Any]) -> dict[str, Any]:
    """
    Safety unwrap: handles both bare Observation and legacy wrapped
    {"observation": {...}, ...} responses regardless of server version.
    """
    if "pr_title" not in data and "observation" in data:
        return data["observation"]
    return data


# ---------------------------------------------------------------------------
# Prompt builders — tight, specific prompts reduce false positives
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    return (
        "You are a senior software security engineer doing a GitHub pull request review.\n"
        "Analyse the code diffs shown and identify ONLY real, concrete issues — do NOT invent problems.\n\n"
        "For EACH issue you find, respond with EXACTLY this JSON (no other text):\n"
        "{\n"
        '  "file": "<exact filename from the diff>",\n'
        '  "line": null,\n'
        '  "issue_type": "<bug|logic|performance|security>",\n'
        '  "comment": "<precise description referencing the specific variable/function/line>",\n'
        '  "suggestion": "<concrete fix>",\n'
        '  "done": false\n'
        "}\n\n"
        "Rules:\n"
        "- ONE issue per response\n"
        "- Only flag issues VISIBLE in the diff (added/changed lines)\n"
        "- Do NOT repeat issues you have already flagged\n"
        "- Set done=true ONLY when you have no more NEW issues to report\n"
        "- Never output anything outside the JSON object\n\n"
        "High-value issue categories to look for:\n"
        "  security: SQL injection, hardcoded secrets, missing auth, path traversal, RCE, weak crypto\n"
        "  bug: None/null dereference, wrong conditionals, missing error handling\n"
        "  logic: wrong formula, silent breaking change, off-by-one\n"
        "  performance: full table scan, unbounded cache, N+1 query"
    )


def _build_user_prompt(obs: dict[str, Any]) -> str:
    """Format the observation as a focused prompt."""
    lines = [
        f"## Pull Request: {obs['pr_title']}",
        f"{obs['pr_description']}",
        "",
        f"[Step {obs['step_count']} of {obs['max_steps']} — find ONE new issue]",
        "",
        "## Changed Files (review only the AFTER code for issues introduced by this PR):",
        "",
    ]

    for diff in obs.get("file_diffs", []):
        lines.append(f"### {diff['filename']} ({diff['language']})")
        lines.append("```")
        lines.append(diff["after"].strip())
        lines.append("```")
        lines.append("")

    prev = obs.get("previous_comments", [])
    if prev:
        lines.append("## Issues you have ALREADY reported (do NOT repeat these):")
        for c in prev:
            lines.append(f"  - [{c['issue_type']}] {c['file']}: {c['comment'][:100]}")
        lines.append("")
        lines.append("Report ONE NEW issue not in the list above, or set done=true if none remain.")
    else:
        lines.append("No issues reported yet. Report the most critical issue you see.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call with retry on transient errors (rate limits, timeouts, 5xx)
# ---------------------------------------------------------------------------

def _is_credit_exhausted(exc: Exception) -> bool:
    """Return True if the error indicates depleted API credits (402 / billing)."""
    msg = str(exc).lower()
    return "402" in msg or "depleted" in msg or "credits" in msg or "billing" in msg


def _call_llm(messages: list[dict[str, str]]) -> str:
    """
    Call the LLM with automatic retry on transient errors.
    Raises immediately on credit-exhausted (402) errors — no point retrying.
    """
    if not HF_TOKEN or HF_TOKEN == "dummy":
    # fallback: return done=true so env continues safely
        return json.dumps({
            "file": "none",
            "line": None,
            "issue_type": "other",
            "comment": "No API token, fallback mode",
            "suggestion": None,
            "done": True
        })
    last_exc: Exception | None = None
    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=400,
                temperature=0.0,   # deterministic / reproducible
                timeout=REQUEST_TIMEOUT,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            if _is_credit_exhausted(exc):
                raise  # no point retrying a billing error
            if attempt < MAX_LLM_RETRIES:
                time.sleep(RETRY_WAIT * (attempt + 1))  # back-off: 5s, 10s
    raise last_exc  # type: ignore[misc]


def _parse_action(raw: str) -> dict[str, Any]:
    """
    Parse LLM output into an action dict.
    Tries several JSON extraction strategies before falling back to done=true.
    """
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]).rstrip("`").strip()

    # Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to extract first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback — signal done to stop the episode cleanly
    return {
        "file": "unknown",
        "line": None,
        "issue_type": "other",
        "comment": text[:200] if text else "Parse failure",
        "suggestion": None,
        "done": True,
    }


# ---------------------------------------------------------------------------
# Score recovery — extract best available score from env state
# ---------------------------------------------------------------------------

def _recover_score(env: Any, rewards: list[float]) -> float:
    """
    Try to recover a meaningful score even when the episode ended abnormally
    (e.g. API credit exhausted mid-task). Uses env.state() if available,
    otherwise falls back to the cumulative reward signal.
    """
    try:
        state = env.state()
        # state() may be a dict or Pydantic object
        if isinstance(state, dict):
            fs = state.get("final_score")
            if fs is not None:
                return float(fs)
            # Compute from issues_found / total
            found = len(state.get("issues_found", []))
            total = state.get("info", {}).get("total_issues", 1)
            fp = state.get("false_positives", 0)
            raw = (found / max(total, 1)) - (fp * 0.1)
            return max(0.0, min(1.0, raw))
        else:
            fs = getattr(state, "final_score", None)
            if fs is not None:
                return float(fs)
            found = len(getattr(state, "issues_found", []))
            total = getattr(getattr(state, "info", {}), "get", lambda k, d: d)("total_issues", 1)
            fp = getattr(state, "false_positives", 0)
            raw = (found / max(total, 1)) - (fp * 0.1)
            return max(0.0, min(1.0, raw))
    except Exception:
        pass

    # Last resort: estimate from reward signal
    # Positive rewards ≈ 0.2 or 0.3 per correct finding
    pos_rewards = sum(r for r in rewards if r > 0)
    estimated = min(1.0, pos_rewards / 1.5)  # rough normalisation
    return round(estimated, 3)


# ---------------------------------------------------------------------------
# Single-task episode runner
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> dict[str, Any]:
    """
    Run one complete episode for the given task.
    Returns a summary dict with steps, score, and rewards.
    """
    env = _build_env_client(task_id)
    obs = _unwrap_obs(env.reset())

    # ── [START] log ─────────────────────────────────────────────────────────
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    rewards: list[float] = []
    step_count = 0
    final_score = 0.0
    success = False
    done = False
    credit_exhausted = False

    # Track best cumulative score seen across all steps — used if episode aborts
    best_cumulative: float = 0.0

    conversation: list[dict[str, str]] = [
        {"role": "system", "content": _build_system_prompt()},
    ]

    while not done and step_count < MAX_STEPS_PER_TASK:
        step_count += 1
        error_str = "null"
        action_str = ""
        reward_val = 0.0

        try:
            user_msg = _build_user_prompt(obs)
            conversation.append({"role": "user", "content": user_msg})

            raw_response = _call_llm(conversation)
            conversation.append({"role": "assistant", "content": raw_response})

            action_dict = _parse_action(raw_response)
            action_str = action_dict.get("comment", "")[:120]

            obs, reward, done, info = env.step(action_dict)
            obs = _unwrap_obs(obs)

            # Normalise reward (dict from DirectEnvClient, Pydantic from DockerEnvClient)
            if isinstance(reward, dict):
                reward_val = reward.get("value", 0.0)
                cumulative_score = float(reward.get("cumulative_score", 0.0))
            else:
                reward_val = float(getattr(reward, "value", 0.0))
                cumulative_score = float(getattr(reward, "cumulative_score", 0.0))

            rewards.append(reward_val)
            # Keep track of the best score we've seen — rescue value if episode aborts
            best_cumulative = max(best_cumulative, cumulative_score)

            if done:
                fs = info.get("final_score")
                if fs is not None:
                    final_score = float(fs)
                else:
                    final_score = cumulative_score
                success = True

        except Exception as exc:
            error_str = str(exc)[:200] if exc else "null"
            done = True

            if _is_credit_exhausted(exc):
                credit_exhausted = True
                # Recover whatever score we earned before credits ran out
                final_score = _recover_score(env, rewards)
                # Episode had real progress — mark success if we scored anything
                success = final_score > 0.0
            else:
                # Other error: still try to recover score from env state
                final_score = _recover_score(env, rewards)
                success = final_score > 0.0

        # ── [STEP] log ───────────────────────────────────────────────────────
        print(
            f"[STEP] step={step_count} action={action_str!r} "
            f"reward={reward_val} done={done} error={error_str}",
            flush=True,
        )

    # If we hit MAX_STEPS_PER_TASK before env signalled done, recover score
    if not done or (not success and rewards):
        recovered = _recover_score(env, rewards)
        if recovered > final_score:
            final_score = recovered
        if final_score > 0.0:
            success = True

    if credit_exhausted:
        print(
            f"[WARN] task={task_id} — HuggingFace credits exhausted at step {step_count}. "
            f"Recovered score={final_score:.3f} from completed steps.",
            file=sys.stderr, flush=True,
        )

    # ── [END] log ────────────────────────────────────────────────────────────
    print(
        f"[END] success={success} steps={step_count} "
        f"score={final_score} rewards={rewards}",
        flush=True,
    )

    return {
        "task_id": task_id,
        "success": success,
        "steps": step_count,
        "score": final_score,
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Main — run all three tasks sequentially
# ---------------------------------------------------------------------------

def main() -> None:
    start_time = time.time()
    all_results: list[dict[str, Any]] = []

    for task_id in TASKS:
        result = run_task(task_id)
        all_results.append(result)

        elapsed = time.time() - start_time
        if elapsed > 18 * 60:
            print(
                f"[WARN] Approaching 20-minute limit ({elapsed:.0f}s elapsed). "
                "Skipping remaining tasks.",
                file=sys.stderr, flush=True,
            )
            break

    avg_score = sum(r["score"] for r in all_results) / max(len(all_results), 1)
    print(f"\n=== AGGREGATE RESULTS ===", flush=True)
    for r in all_results:
        print(
            f"  {r['task_id']}: score={r['score']:.3f} steps={r['steps']} success={r['success']}",
            flush=True,
        )
    print(f"  average_score={avg_score:.3f}", flush=True)
    print(f"  total_elapsed={time.time() - start_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
