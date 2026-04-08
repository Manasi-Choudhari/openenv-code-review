"""
GitHub PR Code Review Environment — OpenEnv Implementation.

Implements:
  reset()  → Observation
  step()   → (Observation, Reward, done, info)
  state()  → EpisodeState
"""

from __future__ import annotations
from typing import Any

from app.models.schemas import (
    Action,
    EpisodeState,
    Observation,
    Reward,
    ReviewComment,
)
from app.tasks.definitions import TASKS, TaskDefinition
from app.graders.grader import grade_action, final_score


class PRReviewEnvironment:
    """
    Simulates a GitHub pull request code review session.

    The agent reads diffs and posts review comments identifying bugs,
    security issues, and performance problems.  A deterministic grader
    scores each comment against a hidden ground truth and returns a
    non-binary reward signal.
    """

    def __init__(self, task_id: str = "task1") -> None:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASKS)}")
        self._task_id = task_id
        self._task: TaskDefinition = TASKS[task_id]

        # Episode state — initialised properly in reset()
        self._step_count: int = 0
        self._issues_found: list[str] = []
        self._false_positives: int = 0
        self._all_comments: list[ReviewComment] = []
        self._done: bool = False
        self._final_score: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment to the start of an episode."""
        self._step_count = 0
        self._issues_found = []
        self._false_positives = 0
        self._all_comments = []
        self._done = False
        self._final_score = None
        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        """
        Apply one agent action (review comment) and return the result.

        Returns:
            observation  — updated observation (same PR, more history)
            reward       — non-binary reward signal
            done         — whether the episode has ended
            info         — auxiliary diagnostic information
        """
        if self._done:
            # Episode already finished — return terminal state with zero reward
            obs = self._build_observation()
            reward = Reward(
                value=0.0,
                reason="Episode already finished.",
                correct_issues_found=len(self._issues_found),
                total_issues=len(self._task.ground_truth),
                false_positives=self._false_positives,
                cumulative_score=self._final_score or 0.0,
            )
            return obs, reward, True, {"message": "Episode already done."}

        self._step_count += 1

        # Record the comment
        comment = ReviewComment(
            file=action.file,
            line=action.line,
            issue_type=action.issue_type,
            comment=action.comment,
            suggestion=action.suggestion,
        )
        self._all_comments.append(comment)

        # Grade the action
        reward, self._issues_found, self._false_positives = grade_action(
            action=action,
            task=self._task,
            already_found=self._issues_found,
            false_positives=self._false_positives,
        )

        # Check episode end conditions
        all_found = len(self._issues_found) == len(self._task.ground_truth)
        max_reached = self._step_count >= self._task.max_steps
        agent_done = action.done

        self._done = all_found or max_reached or agent_done

        info: dict[str, Any] = {
            "step": self._step_count,
            "issues_found": list(self._issues_found),
            "false_positives": self._false_positives,
            "all_issues_found": all_found,
            "max_steps_reached": max_reached,
            "agent_signalled_done": agent_done,
        }

        if self._done:
            self._final_score = final_score(
                task=self._task,
                found_issue_ids=self._issues_found,
                false_positives=self._false_positives,
            )
            info["final_score"] = self._final_score

        obs = self._build_observation()
        return obs, reward, self._done, info

    def state(self) -> EpisodeState:
        """Return the full internal episode state."""
        return EpisodeState(
            task_id=self._task_id,
            pr_title=self._task.pr_title,
            step_count=self._step_count,
            max_steps=self._task.max_steps,
            issues_found=list(self._issues_found),
            false_positives=self._false_positives,
            all_comments=list(self._all_comments),
            done=self._done,
            final_score=self._final_score,
            info={
                "difficulty": self._task.difficulty,
                "total_issues": len(self._task.ground_truth),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        """Construct the current observation from task data and episode state."""
        return Observation(
            pr_title=self._task.pr_title,
            pr_description=self._task.pr_description,
            file_diffs=self._task.file_diffs,
            previous_comments=list(self._all_comments),
            step_count=self._step_count,
            max_steps=self._task.max_steps,
            task_id=self._task_id,
        )
