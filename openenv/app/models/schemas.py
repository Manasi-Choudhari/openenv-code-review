"""
Typed Pydantic models for the GitHub PR Code Review Environment.
Defines Observation, Action, and Reward schemas.
"""

from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class IssueType(str, Enum):
    """Enumeration of possible code issue types an agent can identify."""
    BUG = "bug"
    LOGIC = "logic"
    PERFORMANCE = "performance"
    SECURITY = "security"
    STYLE = "style"
    OTHER = "other"


class FileDiff(BaseModel):
    """Represents a single file's diff within a pull request."""
    filename: str = Field(..., description="Path/name of the file being changed")
    language: str = Field(..., description="Programming language of the file")
    before: str = Field(..., description="Code content before the change")
    after: str = Field(..., description="Code content after the change")
    diff_lines: list[str] = Field(default_factory=list, description="Unified diff lines")


class ReviewComment(BaseModel):
    """A single review comment made by the agent."""
    file: str = Field(..., description="File the comment is about")
    line: int | None = Field(None, description="Line number (optional)")
    issue_type: IssueType = Field(..., description="Type of issue identified")
    comment: str = Field(..., description="Human-readable description of the issue")
    suggestion: str | None = Field(None, description="Optional fix suggestion")


class Observation(BaseModel):
    """
    What the agent sees at each step.
    Contains the PR context, diffs, and history of prior comments.
    """
    pr_title: str = Field(..., description="Title of the pull request")
    pr_description: str = Field(..., description="Description/body of the pull request")
    file_diffs: list[FileDiff] = Field(..., description="List of file diffs in this PR")
    previous_comments: list[ReviewComment] = Field(
        default_factory=list,
        description="Comments the agent has already made this episode"
    )
    step_count: int = Field(0, description="How many steps have been taken so far")
    max_steps: int = Field(10, description="Maximum steps allowed in this episode")
    task_id: str = Field(..., description="Which task is currently active (task1/task2/task3)")


class Action(BaseModel):
    """
    What the agent does at each step: post a review comment.
    """
    file: str = Field(..., description="File being commented on")
    line: int | None = Field(None, description="Line number being flagged (optional)")
    issue_type: IssueType = Field(..., description="Category of issue")
    comment: str = Field(..., description="Description of the issue found")
    suggestion: str | None = Field(None, description="Optional suggested fix")
    done: bool = Field(False, description="Set True when agent believes review is complete")


class Reward(BaseModel):
    """
    Non-binary reward signal for the agent's action.
    Provides partial credit and penalises false positives.
    """
    value: float = Field(..., description="Reward value (can be negative for penalties)")
    reason: str = Field(..., description="Human-readable explanation of the reward")
    correct_issues_found: int = Field(0, description="Count of correctly identified issues so far")
    total_issues: int = Field(..., description="Total issues in the ground truth")
    false_positives: int = Field(0, description="Count of false positives so far")
    cumulative_score: float = Field(0.0, description="Running episode score 0.0–1.0")


class EpisodeState(BaseModel):
    """Internal state of a running episode (returned by state())."""
    task_id: str
    pr_title: str
    step_count: int
    max_steps: int
    issues_found: list[str] = Field(default_factory=list)
    false_positives: int = 0
    all_comments: list[ReviewComment] = Field(default_factory=list)
    done: bool = False
    final_score: float | None = None
    info: dict[str, Any] = Field(default_factory=dict)
