"""Speech chunking configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.config import (
    OMNILINGUAL_CHUNK_MAX_SECONDS,
    OMNILINGUAL_VAD_MIN_SILENCE_MS,
    OMNILINGUAL_VAD_MIN_SPEECH_MS,
    OMNILINGUAL_VAD_NEG_THRESHOLD,
    OMNILINGUAL_VAD_SPEECH_PAD_MS,
    OMNILINGUAL_VAD_THRESHOLD,
)


@dataclass(frozen=True)
class ChunkingConfig:
    """All parameters that govern VAD-based speech chunking."""

    threshold: float
    neg_threshold: float | None
    min_speech_ms: int
    min_silence_ms: int
    speech_pad_ms: int
    chunk_max_seconds: float

    @classmethod
    def from_env(cls) -> "ChunkingConfig":
        return cls(
            threshold=OMNILINGUAL_VAD_THRESHOLD,
            neg_threshold=OMNILINGUAL_VAD_NEG_THRESHOLD,
            min_speech_ms=OMNILINGUAL_VAD_MIN_SPEECH_MS,
            min_silence_ms=OMNILINGUAL_VAD_MIN_SILENCE_MS,
            speech_pad_ms=OMNILINGUAL_VAD_SPEECH_PAD_MS,
            chunk_max_seconds=OMNILINGUAL_CHUNK_MAX_SECONDS,
        )

    def with_overrides(
        self,
        *,
        threshold: float | None = None,
        neg_threshold: float | None = None,
        min_speech_ms: int | None = None,
        min_silence_ms: int | None = None,
        speech_pad_ms: int | None = None,
        chunk_max_seconds: float | None = None,
        neg_threshold_explicit: bool = False,
    ) -> "ChunkingConfig":
        """Return a copy with non-None values replaced.

        `neg_threshold` is special: passing None can either mean "skip override"
        (the default) or "explicitly clear it" (set neg_threshold_explicit=True).
        """
        updates: dict[str, object] = {}
        if threshold is not None:
            updates["threshold"] = threshold
        if neg_threshold is not None or neg_threshold_explicit:
            updates["neg_threshold"] = neg_threshold
        if min_speech_ms is not None:
            updates["min_speech_ms"] = min_speech_ms
        if min_silence_ms is not None:
            updates["min_silence_ms"] = min_silence_ms
        if speech_pad_ms is not None:
            updates["speech_pad_ms"] = speech_pad_ms
        if chunk_max_seconds is not None:
            updates["chunk_max_seconds"] = chunk_max_seconds
        return replace(self, **updates)
