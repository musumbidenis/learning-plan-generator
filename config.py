"""Configuration lookup shared by every module that needs a secret or setting.

One rule, applied everywhere: a value is read from the process environment
(populated once from `.env` locally), and failing that from Streamlit secrets
(which is how Streamlit Community Cloud supplies them - there is no `.env` up
there). Nothing is ever hard-coded in source or committed to the repo.

`ai_client` and the Google Drive layer both go through `get()`, so there is a
single place that knows how configuration is resolved.
"""

from __future__ import annotations

import os

_ENV_LOADED = False


def ensure_env_loaded() -> None:
    """Load `.env` into os.environ exactly once per process (idempotent)."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        from dotenv import load_dotenv
        load_dotenv()
        _ENV_LOADED = True


def get(name: str) -> str:
    """Read a config value from the environment / `.env`, then Streamlit secrets.

    Returns '' when the value is unset anywhere; callers decide whether that is
    an error.
    """
    ensure_env_loaded()
    val = os.getenv(name, "").strip()
    if val:
        return val
    try:                                   # only present when running under Streamlit
        import streamlit as st
        return str(st.secrets.get(name, "")).strip()
    except Exception:                      # noqa: BLE001 - no secrets file, not an error
        return ""
