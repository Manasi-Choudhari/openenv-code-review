"""
Task definitions for the GitHub PR Code Review Environment.

Each task provides:
  - PR metadata (title, description)
  - File diffs with realistic before/after code
  - Ground truth issues for grading
"""

from __future__ import annotations
from dataclasses import dataclass, field
from app.models.schemas import FileDiff, IssueType


@dataclass
class GroundTruthIssue:
    """A single known issue in the PR that the agent should find."""
    issue_id: str                   # Unique stable identifier
    file: str                       # Which file
    issue_type: IssueType           # Category
    description: str                # What the issue is
    keywords: list[str]             # Keywords that must appear in agent comment to match
    is_critical: bool = False       # Critical issues give higher reward


@dataclass
class TaskDefinition:
    """Complete definition of one task."""
    task_id: str
    difficulty: str                         # easy / medium / hard
    pr_title: str
    pr_description: str
    file_diffs: list[FileDiff]
    ground_truth: list[GroundTruthIssue]
    max_steps: int = 10


# ---------------------------------------------------------------------------
# TASK 1 — EASY: Simple syntax / obvious bugs
# ---------------------------------------------------------------------------

TASK1 = TaskDefinition(
    task_id="task1",
    difficulty="easy",
    pr_title="Fix user authentication flow",
    pr_description=(
        "This PR updates the login function to handle edge cases "
        "and refactors the password comparison logic."
    ),
    file_diffs=[
        FileDiff(
            filename="auth/login.py",
            language="python",
            before='''\
def login(username, password):
    user = db.query("SELECT * FROM users WHERE username = ?", username)
    if user and user.password == password:
        return {"status": "ok", "token": generate_token(user.id)}
    return {"status": "fail"}

def generate_token(user_id):
    import random
    return str(random.randint(1000, 9999))
''',
            after='''\
def login(username, password):
    user = db.query("SELECT * FROM users WHERE username = ?", username)
    if user.password == password:          # BUG: no None check
        return {"status": "ok", "token": generate_token(user.id)}
    return {"status": "fail"}

def generate_token(user_id):
    import random                          # BUG: weak random for token
    return str(random.randint(1000, 9999))
''',
            diff_lines=[
                "-    if user and user.password == password:",
                "+    if user.password == password:          # BUG: no None check",
            ],
        ),
        FileDiff(
            filename="auth/utils.py",
            language="python",
            before='''\
def validate_email(email):
    return "@" in email

def hash_password(password):
    import hashlib
    return hashlib.md5(password.encode()).hexdigest()
''',
            after='''\
def validate_email(email):
    return "@" in email                    # BUG: trivially weak validation

def hash_password(password):
    import hashlib
    return hashlib.md5(password.encode()).hexdigest()   # BUG: MD5 is not secure
''',
            diff_lines=[
                "+    return \"@\" in email                    # BUG: trivially weak validation",
                "+    return hashlib.md5(password.encode()).hexdigest()   # BUG: MD5 is not secure",
            ],
        ),
    ],
    ground_truth=[
        GroundTruthIssue(
            issue_id="t1_none_check",
            file="auth/login.py",
            issue_type=IssueType.BUG,
            description="Missing None/null check for `user` before accessing `user.password` causes AttributeError when user is not found.",
            keywords=["none", "null", "attributeerror", "user", "check"],
            is_critical=False,
        ),
        GroundTruthIssue(
            issue_id="t1_weak_token",
            file="auth/login.py",
            issue_type=IssueType.SECURITY,
            description="Token generated with random.randint is cryptographically weak and easily guessable.",
            keywords=["random", "token", "weak", "crypto", "secure", "predictable"],
            is_critical=True,
        ),
        GroundTruthIssue(
            issue_id="t1_md5_password",
            file="auth/utils.py",
            issue_type=IssueType.SECURITY,
            description="MD5 is a broken hash algorithm and must not be used for passwords. Use bcrypt or argon2.",
            keywords=["md5", "hash", "password", "broken", "bcrypt", "argon"],
            is_critical=True,
        ),
        GroundTruthIssue(
            issue_id="t1_email_validation",
            file="auth/utils.py",
            issue_type=IssueType.BUG,
            description="Email validation only checks for '@' — invalid emails like 'a@' or '@b' pass through.",
            keywords=["email", "validation", "regex", "weak", "trivial"],
            is_critical=False,
        ),
    ],
    max_steps=8,
)


# ---------------------------------------------------------------------------
# TASK 2 — MEDIUM: Logic and performance bugs
# ---------------------------------------------------------------------------

TASK2 = TaskDefinition(
    task_id="task2",
    difficulty="medium",
    pr_title="Optimise product search and order processing",
    pr_description=(
        "Refactors product search to support filters and updates "
        "order total calculation for discount codes."
    ),
    file_diffs=[
        FileDiff(
            filename="store/search.py",
            language="python",
            before='''\
def search_products(query, filters=None):
    results = []
    all_products = db.get_all_products()        # loads entire table
    for product in all_products:
        if query.lower() in product.name.lower():
            results.append(product)
    return results
''',
            after='''\
def search_products(query, filters=None):
    results = []
    all_products = db.get_all_products()        # PERF: still loads entire table into memory
    for product in all_products:
        if query.lower() in product.name.lower():
            results.append(product)
    if filters:
        for key, value in filters.items():
            results = [p for p in results if getattr(p, key) == value]
    return results                              # LOGIC: filter applied after loop (correct but O(N) per filter)
''',
            diff_lines=[
                "+    if filters:",
                "+        for key, value in filters.items():",
                "+            results = [p for p in results if getattr(p, key) == value]",
            ],
        ),
        FileDiff(
            filename="store/orders.py",
            language="python",
            before='''\
def calculate_total(cart_items, discount_code=None):
    total = sum(item.price * item.quantity for item in cart_items)
    if discount_code:
        discount = db.get_discount(discount_code)
        total = total - discount.amount
    return total
''',
            after='''\
def calculate_total(cart_items, discount_code=None):
    total = sum(item.price * item.quantity for item in cart_items)
    if discount_code:
        discount = db.get_discount(discount_code)
        total = total * (1 - discount.percent / 100)  # LOGIC: mixed flat+percent discount — wrong formula
    if total < 0:                                      # LOGIC: negative total not prevented before this
        total = 0
    return total
''',
            diff_lines=[
                "-        total = total - discount.amount",
                "+        total = total * (1 - discount.percent / 100)  # changed discount type silently",
                "+    if total < 0:",
                "+        total = 0",
            ],
        ),
        FileDiff(
            filename="store/cache.py",
            language="python",
            before='''\
_cache = {}

def get_cached(key):
    return _cache.get(key)

def set_cached(key, value):
    _cache[key] = value
''',
            after='''\
_cache = {}

def get_cached(key):
    return _cache.get(key)

def set_cached(key, value):
    _cache[key] = value                   # PERF: unbounded in-memory cache — memory leak risk
''',
            diff_lines=[
                "+    _cache[key] = value                   # PERF: unbounded cache",
            ],
        ),
    ],
    ground_truth=[
        GroundTruthIssue(
            issue_id="t2_full_table_scan",
            file="store/search.py",
            issue_type=IssueType.PERFORMANCE,
            description="db.get_all_products() loads the entire products table into memory; use a filtered DB query instead.",
            keywords=["all_products", "full table", "memory", "query", "database", "index", "performance"],
            is_critical=False,
        ),
        GroundTruthIssue(
            issue_id="t2_discount_logic",
            file="store/orders.py",
            issue_type=IssueType.LOGIC,
            description="Discount changed from flat amount to percentage without updating callers — silent breaking change; mixed logic.",
            keywords=["discount", "percent", "amount", "logic", "breaking", "total"],
            is_critical=True,
        ),
        GroundTruthIssue(
            issue_id="t2_negative_total",
            file="store/orders.py",
            issue_type=IssueType.LOGIC,
            description="Negative total check is placed after calculation but a 100%+ discount can still produce negative intermediate values that affect tax calculations upstream.",
            keywords=["negative", "total", "order", "discount", "overflow"],
            is_critical=False,
        ),
        GroundTruthIssue(
            issue_id="t2_unbounded_cache",
            file="store/cache.py",
            issue_type=IssueType.PERFORMANCE,
            description="In-memory dict cache has no eviction policy or size limit — will grow unbounded and cause OOM in production.",
            keywords=["cache", "unbounded", "memory", "eviction", "limit", "leak"],
            is_critical=False,
        ),
    ],
    max_steps=10,
)


# ---------------------------------------------------------------------------
# TASK 3 — HARD: Security vulnerabilities + multi-file
# ---------------------------------------------------------------------------

TASK3 = TaskDefinition(
    task_id="task3",
    difficulty="hard",
    pr_title="Add admin API and user data export feature",
    pr_description=(
        "Adds an admin endpoint for querying users and a data export "
        "endpoint that allows users to download their own data as CSV."
    ),
    file_diffs=[
        FileDiff(
            filename="api/admin.py",
            language="python",
            before='''\
# Admin routes — previously required token auth
from flask import Blueprint, request, jsonify

admin_bp = Blueprint("admin", __name__)
''',
            after='''\
from flask import Blueprint, request, jsonify
import sqlite3

admin_bp = Blueprint("admin", __name__)

@admin_bp.route("/admin/users")
def list_users():
    # SECURITY: No authentication check!
    username_filter = request.args.get("username", "")
    conn = sqlite3.connect("app.db")
    # SECURITY: SQL injection via f-string
    query = f"SELECT * FROM users WHERE username LIKE \'%{username_filter}%\'"
    users = conn.execute(query).fetchall()
    return jsonify(users)

@admin_bp.route("/admin/run", methods=["POST"])
def run_command():
    # SECURITY: Unauthenticated OS command execution
    import subprocess
    cmd = request.json.get("cmd", "")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return jsonify({"output": result.stdout})
''',
            diff_lines=[
                "+@admin_bp.route(\"/admin/users\")",
                "+def list_users():",
                "+    username_filter = request.args.get(\"username\", \"\")",
                "+    query = f\"SELECT * FROM users WHERE username LIKE '%{username_filter}%'\"",
                "+    users = conn.execute(query).fetchall()",
                "+@admin_bp.route(\"/admin/run\", methods=[\"POST\"])",
                "+def run_command():",
                "+    cmd = request.json.get(\"cmd\", \"\")",
                "+    result = subprocess.run(cmd, shell=True, ...)",
            ],
        ),
        FileDiff(
            filename="api/export.py",
            language="python",
            before='''\
# User data export
from flask import Blueprint, request, send_file

export_bp = Blueprint("export", __name__)
''',
            after='''\
import os
from flask import Blueprint, request, send_file

export_bp = Blueprint("export", __name__)

@export_bp.route("/export/data")
def export_user_data():
    user_id = request.args.get("user_id")
    # SECURITY: No ownership check — any user can export any other user's data (IDOR)
    filepath = f"/data/exports/{user_id}_export.csv"
    # SECURITY: Path traversal — user_id could be ../../etc/passwd
    return send_file(filepath)
''',
            diff_lines=[
                "+@export_bp.route(\"/export/data\")",
                "+def export_user_data():",
                "+    user_id = request.args.get(\"user_id\")",
                "+    filepath = f\"/data/exports/{user_id}_export.csv\"",
                "+    return send_file(filepath)",
            ],
        ),
        FileDiff(
            filename="config/settings.py",
            language="python",
            before='''\
import os
SECRET_KEY = os.environ.get("SECRET_KEY", "changeme")
DEBUG = False
''',
            after='''\
import os
SECRET_KEY = "hardcoded_secret_key_do_not_use"   # SECURITY: hardcoded secret
DEBUG = True                                       # SECURITY: debug mode left on
DATABASE_URL = "sqlite:///app.db"
ADMIN_PASSWORD = "admin123"                        # SECURITY: hardcoded credential
''',
            diff_lines=[
                "-SECRET_KEY = os.environ.get(\"SECRET_KEY\", \"changeme\")",
                "+SECRET_KEY = \"hardcoded_secret_key_do_not_use\"",
                "-DEBUG = False",
                "+DEBUG = True",
                "+ADMIN_PASSWORD = \"admin123\"",
            ],
        ),
    ],
    ground_truth=[
        GroundTruthIssue(
            issue_id="t3_sql_injection",
            file="api/admin.py",
            issue_type=IssueType.SECURITY,
            description="SQL injection via f-string interpolation in the users query. Must use parameterised queries.",
            keywords=["sql", "injection", "f-string", "parameterised", "query", "format"],
            is_critical=True,
        ),
        GroundTruthIssue(
            issue_id="t3_no_auth_admin",
            file="api/admin.py",
            issue_type=IssueType.SECURITY,
            description="Admin endpoints have no authentication or authorisation checks — anyone can call them.",
            keywords=["auth", "authentication", "authoris", "admin", "unauthenticated", "access control"],
            is_critical=True,
        ),
        GroundTruthIssue(
            issue_id="t3_rce",
            file="api/admin.py",
            issue_type=IssueType.SECURITY,
            description="Unauthenticated remote code execution via subprocess.run with shell=True and user-controlled input.",
            keywords=["rce", "subprocess", "shell", "command", "injection", "execution"],
            is_critical=True,
        ),
        GroundTruthIssue(
            issue_id="t3_idor",
            file="api/export.py",
            issue_type=IssueType.SECURITY,
            description="Insecure Direct Object Reference (IDOR) — no ownership check means any authenticated user can download any other user's data.",
            keywords=["idor", "ownership", "authoris", "user_id", "access", "other user"],
            is_critical=True,
        ),
        GroundTruthIssue(
            issue_id="t3_path_traversal",
            file="api/export.py",
            issue_type=IssueType.SECURITY,
            description="Path traversal via unvalidated user_id — attacker can set user_id=../../etc/passwd to read arbitrary files.",
            keywords=["path traversal", "directory traversal", "user_id", "filepath", "sanitise", "validate"],
            is_critical=True,
        ),
        GroundTruthIssue(
            issue_id="t3_hardcoded_secrets",
            file="config/settings.py",
            issue_type=IssueType.SECURITY,
            description="Hardcoded SECRET_KEY and ADMIN_PASSWORD in source code. Secrets must come from environment variables or a secrets manager.",
            keywords=["hardcoded", "secret", "password", "credential", "env", "environment variable"],
            is_critical=True,
        ),
        GroundTruthIssue(
            issue_id="t3_debug_mode",
            file="config/settings.py",
            issue_type=IssueType.SECURITY,
            description="DEBUG=True left enabled in production config exposes stack traces and the interactive debugger.",
            keywords=["debug", "production", "stack trace", "debugger"],
            is_critical=False,
        ),
    ],
    max_steps=12,
)


TASKS: dict[str, TaskDefinition] = {
    "task1": TASK1,
    "task2": TASK2,
    "task3": TASK3,
}
