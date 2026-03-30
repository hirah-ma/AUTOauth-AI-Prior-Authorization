"""Load variables from FILES/.env (e.g. GEMINI_API_KEY). Safe if python-dotenv is missing."""
from __future__ import annotations

import os
from pathlib import Path


def load() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return

    # Prefer python-dotenv when available
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv(env_path)
        return

    # Minimal fallback parser (KEY=VALUE), ignores comments and blank lines.
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            # Don't clobber an already-defined environment variable
            os.environ.setdefault(key, value)
    except Exception:
        return
