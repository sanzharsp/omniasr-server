"""Per-chunk annotations that flow through the transcription pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.audio import AudioChunk, WordTimestamp


@dataclass
class ChunkAnnotation:
    """All information collected about a single VAD-chunk.

    Mutable on purpose — enrichment passes (LID, smoothing, alignment) add
    fields in-place rather than building new lists.
    """

    chunk: AudioChunk
    text: str
    language: str | None = None
    language_confidence: float | None = None
    language_inherited: bool = False
    words: list[WordTimestamp] = field(default_factory=list)

    def start_seconds(self, sample_rate: int) -> float:
        return self.chunk.logical_start / sample_rate

    def end_seconds(self, sample_rate: int) -> float:
        return self.chunk.logical_end / sample_rate

    def duration_seconds(self, sample_rate: int) -> float:
        return (self.chunk.logical_end - self.chunk.logical_start) / sample_rate

    def word_count(self) -> int:
        return len(self.text.split())
