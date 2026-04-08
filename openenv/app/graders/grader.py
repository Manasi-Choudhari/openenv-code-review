"""
Deterministic graders for each task.

Scoring formula:
  raw = (correct_found / total_issues) - (false_positives * FP_PENALTY)
  score = clamp(raw, 0.0, 1.0)

Per-step rewards:
  +0.3  critical issue correctly identified
  +0.2  non-critical issue correctly identified
  -0.1  false positive
  -0.05 duplicate (already found this issue)
"""

from __future__ import annotations
import re
from app.models.schemas import Action, Reward, IssueType
from app.tasks.definitions import GroundTruthIssue, TaskDefinition

# Penalty per false positive in the final score
FP_PENALTY = 0.1

# Per-step reward constants
REWARD_CRITICAL = 0.3
REWARD_NORMAL = 0.2
REWARD_FALSE_POSITIVE = -0.1
REWARD_DUPLICATE = -0.05


def _matches_issue(action: Action, issue: GroundTruthIssue) -> bool:
    """
    Determines whether an agent action matches a ground truth issue.
    Matching requires:
      1. Same file (or file not specified)
      2. At least one keyword from the issue's keyword list appears in
         the comment (case-insensitive)
    """
    # File check — allow partial match (agent might abbreviate path)
    if action.file and issue.file:
        if not (action.file in issue.file or issue.file in action.file):
            return False

    # Keyword check — comment must mention at least one expected keyword
    comment_lower = (action.comment or "").lower()
    suggestion_lower = (action.suggestion or "").lower()
    combined = comment_lower + " " + suggestion_lower

    for kw in issue.keywords:
        if kw.lower() in combined:
            return True

    return False


def grade_action(
    action: Action,
    task: TaskDefinition,
    already_found: list[str],
    false_positives: int,
) -> tuple[Reward, list[str], int]:
    """
    Grade a single agent action against a task's ground truth.

    Returns:
        reward      — Reward object
        new_found   — updated list of found issue IDs
        new_fp      — updated false positive count
    """
    total = len(task.ground_truth)
    new_found = list(already_found)
    new_fp = false_positives

    # Try to match against any ground truth issue
    matched_issue: GroundTruthIssue | None = None
    for gt_issue in task.ground_truth:
        if _matches_issue(action, gt_issue):
            matched_issue = gt_issue
            break

    if matched_issue is None:
        # False positive
        new_fp += 1
        reward_val = REWARD_FALSE_POSITIVE
        reason = "False positive — comment does not match any known issue."
    elif matched_issue.issue_id in new_found:
        # Duplicate — already found this one
        reward_val = REWARD_DUPLICATE
        reason = f"Duplicate — issue '{matched_issue.issue_id}' was already identified."
    else:
        # New correct finding
        new_found.append(matched_issue.issue_id)
        if matched_issue.is_critical:
            reward_val = REWARD_CRITICAL
            reason = f"Correct critical issue found: {matched_issue.issue_id}"
        else:
            reward_val = REWARD_NORMAL
            reason = f"Correct issue found: {matched_issue.issue_id}"

    # Compute running cumulative score
    raw = (len(new_found) / total) - (new_fp * FP_PENALTY)
    cumulative = max(0.0, min(1.0, raw))

    reward = Reward(
        value=reward_val,
        reason=reason,
        correct_issues_found=len(new_found),
        total_issues=total,
        false_positives=new_fp,
        cumulative_score=cumulative,
    )

    return reward, new_found, new_fp


def final_score(
    task: TaskDefinition,
    found_issue_ids: list[str],
    false_positives: int,
) -> float:
    """
    Compute the final episode score.
    Clamped to [0.0, 1.0].
    """
    total = len(task.ground_truth)
    if total == 0:
        return 1.0
    raw = (len(found_issue_ids) / total) - (false_positives * FP_PENALTY)
    return max(0.0, min(1.0, raw))
