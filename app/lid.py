"""Language identification (LID) for VAD chunks, with neighbor smoothing."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

import numpy as np
import requests

from app.segments import ChunkAnnotation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LanguageScore:
    """Single-language prediction from a LID backend."""

    language: str
    confidence: float


class LanguageDetector(Protocol):
    """Strategy that maps a chunk transcript (and/or audio) to a language."""

    def detect(
        self,
        *,
        text: str,
        audio: np.ndarray | None = None,
        sample_rate: int = 16000,
    ) -> LanguageScore | None: ...


class NullLanguageDetector:
    """Disabled LID — returns nothing. Default backend when LID is off."""

    def detect(
        self,
        *,
        text: str,
        audio: np.ndarray | None = None,
        sample_rate: int = 16000,
    ) -> LanguageScore | None:
        del text, audio, sample_rate
        return None


class FastTextLanguageDetector:
    """fastText `lid.176.bin` based LID over the transcript text.

    Loaded once. Thread-safe via `_lock` because the underlying C++ model is
    not guaranteed re-entrant.
    """

    _LABEL_RE = re.compile(r"^__label__([A-Za-z\-_]+)$")

    def __init__(self, model_path: str) -> None:
        import fasttext

        fasttext.FastText.eprint = lambda *a, **k: None  # silence the C++ banner
        self._model = fasttext.load_model(model_path)
        self._lock = Lock()
        self.model_path = model_path

    def detect(
        self,
        *,
        text: str,
        audio: np.ndarray | None = None,
        sample_rate: int = 16000,
    ) -> LanguageScore | None:
        del audio, sample_rate

        cleaned = text.strip().replace("\n", " ")
        if not cleaned:
            return None

        with self._lock:
            labels, scores = self._model.predict(cleaned, k=1)
        if not labels:
            return None

        match = self._LABEL_RE.match(labels[0])
        if not match:
            return None

        return LanguageScore(language=match.group(1), confidence=float(scores[0]))


class RemoteHTTPLanguageDetector:
    """Delegates LID to an external HTTP service.

    Expected request:  POST {url}  body={"text": "..."}, optional Bearer token
    Expected response: {"lang": "ru", "scores": [{"code": "ru", "probability": 0.37}, ...]}
    """

    def __init__(
        self,
        *,
        url: str,
        token: str | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.url = url
        self.token = token
        self.timeout_seconds = timeout_seconds
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"
        self._lock = Lock()

    def detect(
        self,
        *,
        text: str,
        audio: np.ndarray | None = None,
        sample_rate: int = 16000,
    ) -> LanguageScore | None:
        del audio, sample_rate

        cleaned = text.strip()
        if not cleaned:
            return None

        try:
            with self._lock:
                response = self._session.post(
                    self.url,
                    json={"text": cleaned},
                    timeout=self.timeout_seconds,
                )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Remote LID call failed: %s", exc)
            return None

        language = payload.get("lang")
        if not language:
            return None

        confidence = 0.0
        for entry in payload.get("scores", []):
            if entry.get("code") == language:
                confidence = float(entry.get("probability") or 0.0)
                break

        return LanguageScore(language=str(language), confidence=confidence)


def load_language_detector(
    backend: str,
    *,
    model_path: str | None = None,
    remote_url: str | None = None,
    remote_token: str | None = None,
    remote_timeout_seconds: float = 5.0,
) -> LanguageDetector:
    """Factory: produce a configured detector or a no-op fallback.

    Errors during loading degrade to NullLanguageDetector instead of failing
    server startup — LID is best-effort enrichment.
    """

    backend = (backend or "none").strip().lower()
    if backend in {"", "none", "off", "disabled"}:
        return NullLanguageDetector()

    if backend == "fasttext":
        if not model_path or not os.path.exists(model_path):
            logger.warning(
                "LID_BACKEND=fasttext but model not found at %r; LID disabled.",
                model_path,
            )
            return NullLanguageDetector()
        try:
            detector = FastTextLanguageDetector(model_path)
            logger.info("LID: fastText loaded from %s", model_path)
            return detector
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load fastText LID; LID disabled.")
            return NullLanguageDetector()

    if backend == "remote":
        if not remote_url:
            logger.warning(
                "LID_BACKEND=remote but OMNILINGUAL_LID_URL is empty; LID disabled."
            )
            return NullLanguageDetector()
        try:
            detector = RemoteHTTPLanguageDetector(
                url=remote_url,
                token=remote_token or None,
                timeout_seconds=remote_timeout_seconds,
            )
            logger.info("LID: remote HTTP detector at %s", remote_url)
            return detector
        except Exception:  # noqa: BLE001
            logger.exception("Failed to init remote LID; LID disabled.")
            return NullLanguageDetector()

    logger.warning("Unknown LID_BACKEND=%r; LID disabled.", backend)
    return NullLanguageDetector()


# --------------------------------------------------------------------------
# Nearest-anchor smoothing
#
# A chunk is an "anchor" when it is long enough AND its LID prediction is
# confident enough to trust. Every non-anchor chunk inherits the language of
# the nearest anchor on the timeline. Short or low-confidence chunks therefore
# never decide their own language — they take it from the closest reliable
# neighbor, which mirrors how a single speaker carries one language across a
# string of short replies.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SmoothingConfig:
    """Thresholds for nearest-anchor language inheritance."""

    short_max_seconds: float = 2.0
    """Chunk shorter than this is short (cannot be an anchor)."""

    short_max_words: int = 3
    """Chunk with at most this many words is short (cannot be an anchor)."""

    anchor_min_confidence: float = 0.7
    """Chunk with LID confidence below this is not trusted as an anchor."""

    fallback_language: str | None = None
    """Used only when no anchor exists in the whole audio."""


def smooth_chunk_languages(
    annotations: list[ChunkAnnotation],
    *,
    config: SmoothingConfig,
    sample_rate: int,
) -> None:
    """Promote anchor chunks' language to every non-anchor chunk in-place.

    A chunk's language is overridden when:
      1. It is not an anchor (short or low-confidence), AND
      2. At least one anchor exists somewhere in the audio.

    The donor is the anchor whose midpoint is closest in time. On a tie,
    the earlier anchor wins (speech tends to carry the previous language
    into a short reply, not the next one).

    Anchor chunks are never modified. Sets `language_inherited=True` on
    chunks whose language was changed.
    """

    if not annotations:
        return

    anchor_flags = [_is_anchor(ann, config, sample_rate) for ann in annotations]

    if not any(anchor_flags):
        # No reliable source anywhere — fall back to request language for
        # chunks that had no detected language at all.
        if config.fallback_language is not None:
            for ann in annotations:
                if ann.language is None:
                    ann.language = config.fallback_language
                    ann.language_inherited = True
        return

    for index, ann in enumerate(annotations):
        if anchor_flags[index]:
            continue

        donor = _nearest_anchor(
            annotations,
            anchor_flags,
            center=index,
            sample_rate=sample_rate,
        )
        if donor is not None and ann.language != donor.language:
            ann.language = donor.language
            ann.language_inherited = True
        elif donor is None and ann.language is None and config.fallback_language:
            ann.language = config.fallback_language
            ann.language_inherited = True


def _is_anchor(
    ann: ChunkAnnotation,
    config: SmoothingConfig,
    sample_rate: int,
) -> bool:
    if ann.language is None:
        return False
    if (ann.language_confidence or 0.0) < config.anchor_min_confidence:
        return False
    if ann.duration_seconds(sample_rate) < config.short_max_seconds:
        return False
    if ann.word_count() <= config.short_max_words:
        return False
    return True


def _chunk_midpoint_seconds(ann: ChunkAnnotation, sample_rate: int) -> float:
    return (ann.start_seconds(sample_rate) + ann.end_seconds(sample_rate)) / 2.0


def _nearest_anchor(
    annotations: list[ChunkAnnotation],
    anchor_flags: list[bool],
    *,
    center: int,
    sample_rate: int,
) -> ChunkAnnotation | None:
    """Pick the anchor with the smallest midpoint-distance to `center`.

    On a distance tie, the anchor with the earlier midpoint wins.
    """

    center_mid = _chunk_midpoint_seconds(annotations[center], sample_rate)
    best: ChunkAnnotation | None = None
    best_distance = float("inf")
    best_midpoint = float("inf")

    for j, ann in enumerate(annotations):
        if j == center or not anchor_flags[j]:
            continue
        midpoint = _chunk_midpoint_seconds(ann, sample_rate)
        distance = abs(midpoint - center_mid)
        if distance < best_distance or (
            distance == best_distance and midpoint < best_midpoint
        ):
            best = ann
            best_distance = distance
            best_midpoint = midpoint

    return best
