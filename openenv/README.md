# GitHub PR Code Review Environment

[![openenv](https://img.shields.io/badge/openenv-compatible-blue)](https://openenv.dev)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces)

---

## Overview

This is an **OpenEnv**-compatible reinforcement learning environment that simulates the GitHub pull request code review process.

An agent reads realistic code diffs across multiple files and must identify bugs, security vulnerabilities, logic errors, and performance issues. A deterministic grader provides a **non-binary, partial-credit reward signal** — the agent is rewarded for each correct finding and penalised for false positives.

---

## Motivation

Code review is one of the most important quality gates in software development, yet it is:

- **Time-consuming** — senior engineers spend 10–15% of their time reviewing code
- **Inconsistent** — humans miss critical issues under cognitive load
- **Security-critical** — SQL injection, path traversal, and hardcoded secrets frequently slip through

Training LLM agents on realistic code review tasks could assist developers by automatically flagging common vulnerabilities and bugs, reducing the cognitive burden on human reviewers and improving security posture.

---

## Observation Space

At each step the agent receives an `Observation` object:

| Field | Type | Description |
|---|---|---|
| `pr_title` | `str` | Title of the pull request |
| `pr_description` | `str` | PR body / description |
| `file_diffs` | `list[FileDiff]` | List of changed files with before/after code |
| `previous_comments` | `list[ReviewComment]` | Comments the agent has already made |
| `step_count` | `int` | Steps taken so far |
| `max_steps` | `int` | Maximum steps allowed |
| `task_id` | `str` | Active task identifier |

Each `FileDiff` contains:

| Field | Description |
|---|---|
| `filename` | Path of the changed file |
| `language` | Programming language |
| `before` | Code before the PR change |
| `after` | Code after the PR change |
| `diff_lines` | Unified diff lines |

---

## Action Space

At each step the agent posts one `Action` (review comment):

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `str` | ✅ | File being commented on |
| `line` | `int \| null` | ❌ | Optional line number |
| `issue_type` | enum | ✅ | `bug / logic / performance / security / style / other` |
| `comment` | `str` | ✅ | Description of the issue |
| `suggestion` | `str \| null` | ❌ | Optional fix suggestion |
| `done` | `bool` | ✅ | `true` when review is complete |

---

## Reward Function

The reward is **non-binary** and provides a partial progress signal:

| Event | Reward |
|---|---|
| Critical issue correctly found | **+0.3** |
| Non-critical issue correctly found | **+0.2** |
| False positive (no matching ground-truth issue) | **−0.1** |
| Duplicate (already found this issue) | **−0.05** |

**Episode score** (returned as `cumulative_score`):

```
score = clamp((correct_found / total_issues) − (false_positives × 0.1), 0.0, 1.0)
```

---

## Tasks

### Task 1 — Easy: Authentication Module Bugs

**PR:** "Fix user authentication flow"  
**Files:** `auth/login.py`, `auth/utils.py`  
**Max steps:** 8

The agent must detect four issues:

1. Missing `None` check before accessing `user.password` → `AttributeError`
2. Cryptographically weak token generation using `random.randint`
3. MD5 used for password hashing (broken algorithm)
4. Trivially weak email validation (only checks for `@`)

**Grading:** Deterministic keyword matching against ground-truth issue list.

---

### Task 2 — Medium: E-Commerce Logic & Performance

**PR:** "Optimise product search and order processing"  
**Files:** `store/search.py`, `store/orders.py`, `store/cache.py`  
**Max steps:** 10

The agent must detect four issues:

1. Full table scan on every search request (`db.get_all_products()`)
2. Silent discount type change (flat amount → percentage) — breaking logic change
3. Negative order total possible before guard is applied
4. Unbounded in-memory cache with no eviction policy (memory leak)

---

### Task 3 — Hard: Security Vulnerabilities (Multi-File)

**PR:** "Add admin API and user data export feature"  
**Files:** `api/admin.py`, `api/export.py`, `config/settings.py`  
**Max steps:** 12

The agent must detect seven issues:

1. **SQL injection** — f-string used to build SQL query in admin endpoint
2. **No authentication** — admin endpoints accessible without any auth token
3. **Remote Code Execution (RCE)** — unauthenticated `subprocess.run` with `shell=True`
4. **IDOR** — no ownership check on data export endpoint
5. **Path traversal** — unvalidated `user_id` used in file path construction
6. **Hardcoded secrets** — `SECRET_KEY` and `ADMIN_PASSWORD` in source code
7. **Debug mode on** — `DEBUG=True` left in production config

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (for containerised deployment)

### Local Installation

```bash
git clone <repo>
cd openenv-pr-review

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Start the API Server

```bash
uvicorn server:app --host 0.0.0.0 --port 7860 --reload
```

API docs: http://localhost:7860/docs

### Docker Build & Run

```bash
docker build -t pr-review-env .
docker run -p 7860:7860 pr-review-env
```

---

## Running Inference

Set required environment variables then run `inference.py`:

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token_here"

# Optional — use a local Docker container as the environment
# export LOCAL_IMAGE_NAME="pr-review-env"

python inference.py
```

Expected log output:

```
[START] task=task1 env=PRReviewEnvironment model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action='Missing None check for user before accessing user.password' reward=0.2 done=False error=null
[STEP] step=2 action='MD5 is broken for passwords, use bcrypt' reward=0.3 done=False error=null
...
[END] success=True steps=5 score=0.85 rewards=[0.2, 0.3, 0.3, -0.1, 0.2]
```

---

## Baseline Scores

Scores obtained with `meta-llama/Llama-3.1-8B-Instruct` (temperature=0):

| Task | Difficulty | Baseline Score | Notes |
|---|---|---|---|
| task1 | Easy | **0.75** | Misses weak email validation |
| task2 | Medium | **0.50** | Catches performance issues, misses negative total |
| task3 | Hard | **0.43** | Catches SQL injection and hardcoded secrets; misses IDOR/path traversal |
| **Average** | — | **0.56** | — |

---

## Project Structure

```
openenv-pr-review/
├── app/
│   ├── environment/
│   │   └── env.py          # PRReviewEnvironment (step/reset/state)
│   ├── models/
│   │   └── schemas.py      # Pydantic: Observation, Action, Reward, EpisodeState
│   ├── tasks/
│   │   └── definitions.py  # Task 1/2/3 with diffs and ground truth
│   └── graders/
│       └── grader.py       # Deterministic grader + reward computation
├── server.py               # FastAPI HTTP server (HF Spaces / Docker)
├── inference.py            # Baseline inference script
├── Dockerfile              # Container definition
├── openenv.yaml            # OpenEnv spec file
├── requirements.txt
└── README.md
```

---

## HuggingFace Spaces Deployment

1. Push repository to a HF Space with **Docker SDK**
2. Space will build the image and expose port `7860`
3. The `POST /reset` endpoint will be available immediately

The `openenv.yaml` includes the required `openenv` tag for registry compatibility.

---

## License

MIT
