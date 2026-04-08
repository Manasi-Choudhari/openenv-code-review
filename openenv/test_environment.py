"""
Basic smoke tests for the PR Review Environment.
Run with: python -m pytest tests/ -v
"""

import pytest
from app.environment.env import PRReviewEnvironment
from app.models.schemas import Action, IssueType, Observation, Reward, EpisodeState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_action(file: str, comment: str, issue_type: IssueType = IssueType.BUG, done: bool = False) -> Action:
    return Action(file=file, issue_type=issue_type, comment=comment, done=done)


# ---------------------------------------------------------------------------
# Task 1 tests
# ---------------------------------------------------------------------------

class TestTask1:
    def setup_method(self):
        self.env = PRReviewEnvironment(task_id="task1")

    def test_reset_returns_observation(self):
        obs = self.env.reset()
        assert isinstance(obs, Observation)
        assert obs.task_id == "task1"
        assert obs.step_count == 0
        assert len(obs.file_diffs) == 2

    def test_correct_bug_rewarded(self):
        self.env.reset()
        action = make_action(
            file="auth/login.py",
            comment="Missing None check for user before accessing user.password causes AttributeError",
            issue_type=IssueType.BUG,
        )
        obs, reward, done, info = self.env.step(action)
        assert isinstance(reward, Reward)
        assert reward.value > 0, "Correct finding should get positive reward"
        assert reward.correct_issues_found == 1

    def test_critical_issue_gets_higher_reward(self):
        self.env.reset()
        action = make_action(
            file="auth/login.py",
            comment="random.randint is cryptographically weak for token generation — use secrets module",
            issue_type=IssueType.SECURITY,
        )
        _, reward, _, _ = self.env.step(action)
        assert reward.value == 0.3, "Critical issue should reward 0.3"

    def test_false_positive_penalised(self):
        self.env.reset()
        action = make_action(
            file="auth/login.py",
            comment="This function is poorly named and hard to read",
            issue_type=IssueType.STYLE,
        )
        _, reward, _, _ = self.env.step(action)
        assert reward.value < 0, "False positive should get negative reward"
        assert reward.false_positives == 1

    def test_duplicate_penalised(self):
        self.env.reset()
        action = make_action(
            file="auth/login.py",
            comment="None check missing for user object — AttributeError",
        )
        self.env.step(action)  # First time — correct
        _, reward, _, _ = self.env.step(action)  # Second time — duplicate
        assert reward.value < 0, "Duplicate should be penalised"

    def test_done_when_all_found(self):
        self.env.reset()
        actions = [
            make_action("auth/login.py", "None check missing for user object — AttributeError"),
            make_action("auth/login.py", "Token uses random.randint — cryptographically weak, use secrets", IssueType.SECURITY),
            make_action("auth/utils.py", "MD5 is broken for password hashing, use bcrypt or argon2", IssueType.SECURITY),
            make_action("auth/utils.py", "Email validation only checks for @ — trivially weak regex"),
        ]
        done = False
        for a in actions:
            _, _, done, _ = self.env.step(a)
        assert done, "Episode should end when all 4 issues are found"

    def test_state_method(self):
        self.env.reset()
        state = self.env.state()
        assert isinstance(state, EpisodeState)
        assert state.task_id == "task1"

    def test_max_steps_terminates_episode(self):
        self.env.reset()
        done = False
        for _ in range(10):  # task1 max_steps=8
            _, _, done, _ = self.env.step(
                make_action("auth/login.py", "some comment about nothing specific")
            )
            if done:
                break
        assert done, "Episode must terminate at max_steps"


# ---------------------------------------------------------------------------
# Task 2 tests
# ---------------------------------------------------------------------------

class TestTask2:
    def setup_method(self):
        self.env = PRReviewEnvironment(task_id="task2")

    def test_reset(self):
        obs = self.env.reset()
        assert obs.task_id == "task2"
        assert len(obs.file_diffs) == 3

    def test_performance_issue_detected(self):
        self.env.reset()
        action = make_action(
            file="store/search.py",
            comment="db.get_all_products() loads the full table into memory — use a filtered database query with an index instead",
            issue_type=IssueType.PERFORMANCE,
        )
        _, reward, _, _ = self.env.step(action)
        assert reward.value > 0

    def test_logic_discount_detected(self):
        self.env.reset()
        action = make_action(
            file="store/orders.py",
            comment="Discount changed from flat amount to percentage — breaking change; mixed logic in calculate_total",
            issue_type=IssueType.LOGIC,
        )
        _, reward, _, _ = self.env.step(action)
        assert reward.value > 0


# ---------------------------------------------------------------------------
# Task 3 tests
# ---------------------------------------------------------------------------

class TestTask3:
    def setup_method(self):
        self.env = PRReviewEnvironment(task_id="task3")

    def test_reset(self):
        obs = self.env.reset()
        assert obs.task_id == "task3"
        assert len(obs.file_diffs) == 3

    def test_sql_injection_detected(self):
        self.env.reset()
        action = make_action(
            file="api/admin.py",
            comment="SQL injection via f-string interpolation — use parameterised query instead",
            issue_type=IssueType.SECURITY,
        )
        _, reward, _, _ = self.env.step(action)
        assert reward.value == 0.3  # critical

    def test_rce_detected(self):
        self.env.reset()
        action = make_action(
            file="api/admin.py",
            comment="subprocess.run with shell=True and user-controlled input allows remote code execution",
            issue_type=IssueType.SECURITY,
        )
        _, reward, _, _ = self.env.step(action)
        assert reward.value == 0.3  # critical

    def test_hardcoded_secret_detected(self):
        self.env.reset()
        action = make_action(
            file="config/settings.py",
            comment="Hardcoded SECRET_KEY and ADMIN_PASSWORD must be moved to environment variables",
            issue_type=IssueType.SECURITY,
        )
        _, reward, _, _ = self.env.step(action)
        assert reward.value == 0.3  # critical


# ---------------------------------------------------------------------------
# Unknown task raises error
# ---------------------------------------------------------------------------

def test_invalid_task_raises():
    with pytest.raises(ValueError, match="Unknown task_id"):
        PRReviewEnvironment(task_id="task99")
