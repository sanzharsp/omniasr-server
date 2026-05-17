"""
Configuration constants for Omnilingual-ASR server.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

load_dotenv(dotenv_path=ENV_FILE)

DEFAULT_MODEL_NAME = "omniASR_CTC_300M_v2"
DEFAULT_ZERO_SHOT_MODEL_NAME: str | None = None
DEFAULT_ALIGNMENT_MODEL_NAME: str | None = None
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_ROOT_PATH = ""
DEFAULT_DEVICE = "auto"
DEFAULT_BATCH_SIZE = 4
DEFAULT_PRELOAD_ZERO_SHOT = False
DEFAULT_CHUNK_MAX_SECONDS = 30.0
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_VAD_NEG_THRESHOLD: float | None = None
DEFAULT_VAD_MIN_SPEECH_MS = 250
DEFAULT_VAD_MIN_SILENCE_MS = 300
DEFAULT_VAD_SPEECH_PAD_MS = 200


def _get_int_env(name: str, default: int) -> int:
    """Return an integer environment variable with a sane fallback."""

    value = os.getenv(name)
    if value in (None, ""):
        return default

    return int(value)


def _get_float_env(name: str, default: float) -> float:
    """Return a float environment variable with a sane fallback."""

    value = os.getenv(name)
    if value in (None, ""):
        return default

    return float(value)


def _get_optional_float_env(name: str, default: float | None) -> float | None:
    """Return an optional float environment variable with blank-as-none support."""

    value = os.getenv(name)
    if value in (None, ""):
        return default

    return float(value)


def _get_optional_str_env(name: str, default: str | None) -> str | None:
    """Return an optional string environment variable with blank-as-none support."""

    value = os.getenv(name)
    if value is None:
        return default

    stripped = value.strip()
    if stripped == "":
        return default

    return stripped


def _get_bool_env(name: str, default: bool) -> bool:
    """Return a boolean environment variable with common truthy/falsy values."""

    value = os.getenv(name)
    if value in (None, ""):
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"Invalid {name} value: {value!r}. Use true/false.")


# Available models:
# - omniASR_CTC_{300M,1B,3B,7B}_v2: Fast parallel CTC generation
# - omniASR_LLM_{300M,1B,3B,7B}_v2: Language-conditioned autoregressive
# - omniASR_LLM_Unlimited_{300M,1B,3B,7B}_v2: Unlimited audio length
# Configure zero-shot variants separately via ZERO_SHOT_MODEL_NAME.
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME).strip()
ZERO_SHOT_MODEL_NAME = _get_optional_str_env(
    "ZERO_SHOT_MODEL_NAME",
    DEFAULT_ZERO_SHOT_MODEL_NAME,
)
# Optional CTC model used solely for forced-alignment of word timestamps.
# When unset, falls back to HeuristicWordAligner.
ALIGNMENT_MODEL_NAME = _get_optional_str_env(
    "ALIGNMENT_MODEL_NAME",
    DEFAULT_ALIGNMENT_MODEL_NAME,
)
OMNILINGUAL_HOST = os.getenv("OMNILINGUAL_HOST", DEFAULT_HOST)
OMNILINGUAL_PORT = _get_int_env("OMNILINGUAL_PORT", DEFAULT_PORT)
OMNILINGUAL_ROOT_PATH = os.getenv("OMNILINGUAL_ROOT_PATH", DEFAULT_ROOT_PATH)
OMNILINGUAL_DEVICE = os.getenv("OMNILINGUAL_DEVICE", DEFAULT_DEVICE).lower()
OMNILINGUAL_BATCH_SIZE = _get_int_env("OMNILINGUAL_BATCH_SIZE", DEFAULT_BATCH_SIZE)
OMNILINGUAL_PRELOAD_ZERO_SHOT = _get_bool_env(
    "OMNILINGUAL_PRELOAD_ZERO_SHOT",
    DEFAULT_PRELOAD_ZERO_SHOT,
)
OMNILINGUAL_CHUNK_MAX_SECONDS = _get_float_env(
    "OMNILINGUAL_CHUNK_MAX_SECONDS",
    DEFAULT_CHUNK_MAX_SECONDS,
)
OMNILINGUAL_VAD_THRESHOLD = _get_float_env(
    "OMNILINGUAL_VAD_THRESHOLD",
    DEFAULT_VAD_THRESHOLD,
)
OMNILINGUAL_VAD_NEG_THRESHOLD = _get_optional_float_env(
    "OMNILINGUAL_VAD_NEG_THRESHOLD",
    DEFAULT_VAD_NEG_THRESHOLD,
)
OMNILINGUAL_VAD_MIN_SPEECH_MS = _get_int_env(
    "OMNILINGUAL_VAD_MIN_SPEECH_MS",
    DEFAULT_VAD_MIN_SPEECH_MS,
)
OMNILINGUAL_VAD_MIN_SILENCE_MS = _get_int_env(
    "OMNILINGUAL_VAD_MIN_SILENCE_MS",
    DEFAULT_VAD_MIN_SILENCE_MS,
)
OMNILINGUAL_VAD_SPEECH_PAD_MS = _get_int_env(
    "OMNILINGUAL_VAD_SPEECH_PAD_MS",
    DEFAULT_VAD_SPEECH_PAD_MS,
)

# Per-chunk language identification (LID). Off by default; set
# OMNILINGUAL_LID_BACKEND=fasttext to enable. Smoothing reduces noise from
# short/low-confidence chunks by inheriting language from reliable neighbors.
OMNILINGUAL_LID_BACKEND = os.getenv("OMNILINGUAL_LID_BACKEND", "none").strip().lower()
OMNILINGUAL_LID_MODEL_PATH = os.getenv(
    "OMNILINGUAL_LID_MODEL_PATH",
    "/models/lid/lid.176.bin",
)
OMNILINGUAL_LID_URL = os.getenv("OMNILINGUAL_LID_URL", "").strip()
OMNILINGUAL_LID_TOKEN = os.getenv("OMNILINGUAL_LID_TOKEN", "").strip()
OMNILINGUAL_LID_TIMEOUT_SECONDS = _get_float_env(
    "OMNILINGUAL_LID_TIMEOUT_SECONDS",
    5.0,
)
# Nearest-anchor smoothing: short / low-confidence chunks inherit the language
# of the nearest "anchor" chunk on the timeline.
OMNILINGUAL_LID_SHORT_MAX_SECONDS = _get_float_env(
    "OMNILINGUAL_LID_SHORT_MAX_SECONDS",
    2.0,
)
OMNILINGUAL_LID_SHORT_MAX_WORDS = _get_int_env(
    "OMNILINGUAL_LID_SHORT_MAX_WORDS",
    3,
)
OMNILINGUAL_LID_ANCHOR_MIN_CONFIDENCE = _get_float_env(
    "OMNILINGUAL_LID_ANCHOR_MIN_CONFIDENCE",
    0.7,
)

# Optional bearer-token authentication. When set, /v1/* endpoints require
# `Authorization: Bearer <key>`. /healthz, /readyz, /docs, /openapi.json
# stay open so probes and OpenAPI tooling work.
OMNILINGUAL_API_KEY = os.getenv("OMNILINGUAL_API_KEY", "").strip()
