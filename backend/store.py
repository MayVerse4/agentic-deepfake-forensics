"""
Users, sessions and per-user analysis history in a local SQLite file
(backend/data/app.db -- no database server needed, works offline).

Passwords: scrypt (Python stdlib) with a random per-user salt; the plain
password is never stored. Sessions: a random token sent to the browser in
an httpOnly cookie; only its SHA-256 is stored, so a leaked database file
cannot be used to log in.
"""

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
import uuid

DB_PATH = os.environ.get("APP_DB_PATH", os.path.join(os.path.dirname(__file__), "data", "app.db"))
SESSION_TTL = 7 * 24 * 3600
MIN_PASSWORD = 8
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init():
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
                password_hash TEXT NOT NULL, salt TEXT NOT NULL, created_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at REAL NOT NULL, record TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS analyses_user ON analyses(user_id, created_at);
        """)


def _hash_password(password, salt_hex):
    return hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), dklen=64, **SCRYPT).hex()


def _public(row):
    return {"id": row["id"], "email": row["email"], "name": row["name"]}


# ------------------------------------------------------------------ users
def create_user(email, name, password):
    """Returns the new user; raises ValueError with a user-facing message."""
    email = (email or "").strip().lower()
    name = (name or "").strip() or email.split("@")[0]
    if not EMAIL_RE.match(email):
        raise ValueError("Enter a valid email address.")
    if len(password or "") < MIN_PASSWORD:
        raise ValueError(f"Password must be at least {MIN_PASSWORD} characters.")
    salt = secrets.token_hex(16)
    user = {"id": uuid.uuid4().hex, "email": email, "name": name[:80]}
    try:
        with _conn() as c:
            c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)",
                      (user["id"], email, user["name"], _hash_password(password, salt), salt, time.time()))
    except sqlite3.IntegrityError:
        raise ValueError("An account with this email already exists.")
    return user


def authenticate(email, password):
    """Returns the user, or None for a wrong email or password (same answer for both)."""
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE email = ?", ((email or "").strip().lower(),)).fetchone()
    if row is None:
        _hash_password(password or "", secrets.token_hex(16))  # same cost either way: no timing hint
        return None
    if not hmac.compare_digest(_hash_password(password or "", row["salt"]), row["password_hash"]):
        return None
    return _public(row)


# ------------------------------------------------------------------ sessions
def _token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user_id):
    token = secrets.token_urlsafe(32)
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
        c.execute("INSERT INTO sessions VALUES (?, ?, ?)", (_token_hash(token), user_id, time.time() + SESSION_TTL))
    return token


def user_for_token(token):
    if not token:
        return None
    with _conn() as c:
        row = c.execute("""SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id
                           WHERE s.token_hash = ? AND s.expires_at > ?""",
                        (_token_hash(token), time.time())).fetchone()
    return _public(row) if row else None


def delete_session(token):
    if token:
        with _conn() as c:
            c.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))


# ------------------------------------------------------------------ history
def save_analysis(user_id, record):
    rid = uuid.uuid4().hex
    now = time.time()
    record = {**record, "_id": rid, "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))}
    with _conn() as c:
        c.execute("INSERT INTO analyses VALUES (?, ?, ?, ?)", (rid, user_id, now, json.dumps(record)))
    return record


def list_analyses(user_id):
    with _conn() as c:
        rows = c.execute("SELECT record FROM analyses WHERE user_id = ? ORDER BY created_at DESC", (user_id,)).fetchall()
    return [json.loads(r["record"]) for r in rows]


def delete_analysis(user_id, rid):
    with _conn() as c:
        return c.execute("DELETE FROM analyses WHERE id = ? AND user_id = ?", (rid, user_id)).rowcount > 0
