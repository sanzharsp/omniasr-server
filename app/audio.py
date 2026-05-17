"""
Audio decoding, chunking, and transcript stitching helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import io
import itertools
from pathlib import Path
from typing import BinaryIO

import av
import numpy as np
import torch
from omnilingual_asr.models.inference.pipeline import MAX_ALLOWED_AUDIO_SEC
from silero_vad import get_speech_timestamps

from app.chunking import ChunkingConfig

MODEL_MAX_AUDIO_SECONDS = float(MAX_ALLOWED_AUDIO_SEC)
TARGET_SAMPLE_RATE = 16000

SENTENCE_END_CHARS = frozenset(".!?")
NO_SPACE_BEFORE_CHARS = frozenset(",.!?;:%)]}")
NO_SPACE_AFTER_CHARS = frozenset("([{")


@dataclass(frozen=True)
class AudioChunk:
    """A decoded audio slice represented in sample indices.

    `start_sample`/`end_sample` are the inference window (with VAD padding).
    `logical_start_sample`/`logical_end_sample` are the unpadded VAD bounds
    used for downstream timestamps. When omitted they default to the padded
    bounds (legacy callers and synthetic chunks).
    """

    start_sample: int
    end_sample: int
    logical_start_sample: int | None = None
    logical_end_sample: int | None = None

    @property
    def logical_start(self) -> int:
        return self.start_sample if self.logical_start_sample is None else self.logical_start_sample

    @property
    def logical_end(self) -> int:
        return self.end_sample if self.logical_end_sample is None else self.logical_end_sample

    def duration_seconds(self, sample_rate: int) -> float:
        return (self.end_sample - self.start_sample) / sample_rate


@dataclass(frozen=True)
class WordTimestamp:
    """An approximate word-level timestamp within a decoded audio stream."""

    word: str
    start_seconds: float
    end_seconds: float
    language: str | None = None


@dataclass(frozen=True)
class TimedTranscriptSegment:
    """A transcript segment tied to a speech-aware audio chunk."""

    start_seconds: float
    end_seconds: float
    text: str
    words: list[WordTimestamp]
    language: str | None = None


@dataclass(frozen=True)
class DecodedAudio:
    """A decoded mono waveform ready for chunk planning."""

    waveform: np.ndarray
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return len(self.waveform) / self.sample_rate

    def to_model_input(self, chunk: AudioChunk) -> dict[str, np.ndarray | int]:
        waveform = self.waveform[chunk.start_sample : chunk.end_sample]
        return {
            "waveform": np.ascontiguousarray(waveform, dtype=np.float32),
            "sample_rate": self.sample_rate,
        }


def decode_audio_bytes(
    audio_bytes: bytes,
    *,
    content_type: str | None = None,
    filename: str | None = None,
) -> DecodedAudio:
    """Decode arbitrary audio bytes into mono 16kHz float32 waveform."""

    if content_type in {"audio/pcm", "audio/raw"}:
        waveform = _decode_raw_pcm(audio_bytes)
        return DecodedAudio(waveform=waveform, sample_rate=TARGET_SAMPLE_RATE)

    waveform = _decode_with_pyav(
        io.BytesIO(audio_bytes),
        filename=filename,
        sampling_rate=TARGET_SAMPLE_RATE,
    )
    if waveform.size == 0:
        raise ValueError("Decoded audio is empty.")

    return DecodedAudio(waveform=waveform, sample_rate=TARGET_SAMPLE_RATE)


def build_speech_chunks(
    audio: DecodedAudio,
    *,
    vad_model: torch.nn.Module,
    vad_device: torch.device,
    config: ChunkingConfig,
) -> list[AudioChunk]:
    """Split audio into Silero VAD speech-aware chunks below the model limit."""

    _validate_chunking_options(config)

    max_chunk_samples = int(config.chunk_max_seconds * audio.sample_rate)
    audio_tensor = torch.from_numpy(audio.waveform).to(vad_device)
    speech_segments = get_speech_timestamps(
        audio_tensor,
        vad_model,
        threshold=config.threshold,
        sampling_rate=audio.sample_rate,
        min_speech_duration_ms=config.min_speech_ms,
        max_speech_duration_s=config.chunk_max_seconds,
        min_silence_duration_ms=config.min_silence_ms,
        speech_pad_ms=config.speech_pad_ms,
        neg_threshold=config.neg_threshold,
        return_seconds=False,
    )
    chunks = _speech_timestamps_to_chunks(
        speech_segments,
        total_samples=len(audio.waveform),
        pad_samples=int((config.speech_pad_ms / 1000) * audio.sample_rate),
    )

    if not chunks:
        if len(audio.waveform) <= max_chunk_samples:
            return [AudioChunk(0, len(audio.waveform))]

        return _split_fixed_windows(
            total_samples=len(audio.waveform),
            max_chunk_samples=max_chunk_samples,
        )

    return _merge_adjacent_segments(
        chunks,
        max_chunk_samples=max_chunk_samples,
        sample_rate=audio.sample_rate,
        speech_pad_ms=config.speech_pad_ms,
    )


def join_transcript_texts(texts: list[str]) -> str:
    """Join chunk transcriptions with normalized spacing."""

    parts: list[str] = []
    for text in texts:
        normalized = text.strip()
        if not normalized:
            continue
        if not parts:
            parts.append(normalized)
            continue

        separator = _get_text_separator(parts[-1], normalized)
        parts.append(f"{separator}{normalized}")

    return "".join(parts)


def build_timed_segments(
    chunks: list[AudioChunk],
    texts: list[str],
    *,
    sample_rate: int,
    waveform: np.ndarray | None = None,
    aligner: "WordAlignerLike | None" = None,
) -> list[TimedTranscriptSegment]:
    """Map chunk transcriptions back to chunk times with approximate word timings.

    Segment timestamps use the logical (un-padded) VAD bounds. Word timestamps
    come from `aligner`; the chunk waveform slice is handed to it so that
    model-based aligners can compute alignments from emissions.
    """

    if aligner is None:
        from app.aligners import HeuristicWordAligner

        aligner = HeuristicWordAligner()

    segments: list[TimedTranscriptSegment] = []

    for chunk, text in zip(chunks, texts, strict=True):
        normalized = text.strip()
        if not normalized:
            continue

        start_seconds = chunk.logical_start / sample_rate
        end_seconds = chunk.logical_end / sample_rate

        chunk_waveform = (
            waveform[chunk.start_sample : chunk.end_sample]
            if waveform is not None
            else np.empty(0, dtype=np.float32)
        )

        words = aligner.align(
            text=normalized,
            waveform=chunk_waveform,
            sample_rate=sample_rate,
            chunk_start_seconds=start_seconds,
            chunk_end_seconds=end_seconds,
        )
        segments.append(
            TimedTranscriptSegment(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                text=normalized,
                words=words,
            )
        )

    return segments


def _decode_raw_pcm(audio_bytes: bytes) -> np.ndarray:
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    if audio_int16.size == 0:
        raise ValueError("Decoded audio is empty.")

    return np.ascontiguousarray(audio_int16.astype(np.float32) / 32768.0)


def _decode_with_pyav(
    source: BinaryIO,
    *,
    filename: str | None,
    sampling_rate: int,
) -> np.ndarray:
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=sampling_rate,
    )
    raw_buffer = io.BytesIO()
    dtype: np.dtype | None = None
    format_hint = _guess_container_format(filename)

    try:
        with av.open(
            source,
            mode="r",
            metadata_errors="ignore",
            format=format_hint,
        ) as container:
            frames = container.decode(audio=0)
            frames = _ignore_invalid_frames(frames)
            frames = _group_frames(frames, 500000)
            frames = _resample_frames(frames, resampler)

            for frame in frames:
                array = frame.to_ndarray()
                dtype = array.dtype
                raw_buffer.write(array)
    finally:
        del resampler
        gc.collect()

    if dtype is None:
        return np.array([], dtype=np.float32)

    audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)
    return np.ascontiguousarray(audio.astype(np.float32) / 32768.0)


def _guess_container_format(filename: str | None) -> str | None:
    if not filename:
        return None

    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix == "m4a":
        return "ipod"
    return None


def _ignore_invalid_frames(frames):
    iterator = iter(frames)

    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue


def _group_frames(frames, num_samples: int | None = None):
    fifo = av.audio.fifo.AudioFifo()

    for frame in frames:
        frame.pts = None
        fifo.write(frame)

        if num_samples is not None and fifo.samples >= num_samples:
            yield fifo.read()

    if fifo.samples > 0:
        yield fifo.read()


def _resample_frames(frames, resampler):
    for frame in itertools.chain(frames, [None]):
        yield from resampler.resample(frame)


def _validate_chunking_options(config: ChunkingConfig) -> None:
    if not 0 < config.chunk_max_seconds < MODEL_MAX_AUDIO_SECONDS:
        raise ValueError(
            f"`chunk_max_seconds` must be between 0 and {MODEL_MAX_AUDIO_SECONDS}."
        )
    if not 0.0 <= config.threshold <= 1.0:
        raise ValueError("`threshold` must be between 0 and 1.")
    if config.neg_threshold is not None and not 0.0 <= config.neg_threshold <= 1.0:
        raise ValueError("`neg_threshold` must be between 0 and 1.")
    if config.min_speech_ms < 0 or config.min_silence_ms < 0 or config.speech_pad_ms < 0:
        raise ValueError("VAD timing options must be non-negative.")


def _speech_timestamps_to_chunks(
    speech_segments: list[dict[str, int]],
    *,
    total_samples: int,
    pad_samples: int,
) -> list[AudioChunk]:
    chunks: list[AudioChunk] = []

    for segment in speech_segments:
        start_sample = max(0, int(segment["start"]))
        end_sample = min(total_samples, int(segment["end"]))

        if end_sample <= start_sample:
            continue

        # Silero inflates [start, end] outward by speech_pad_ms. Recover the
        # un-padded VAD bounds so downstream timestamps don't drift by pad_ms.
        logical_start = min(end_sample - 1, start_sample + pad_samples)
        logical_end = max(logical_start + 1, end_sample - pad_samples)

        chunks.append(
            AudioChunk(
                start_sample=start_sample,
                end_sample=end_sample,
                logical_start_sample=logical_start,
                logical_end_sample=logical_end,
            )
        )

    return chunks


def _split_fixed_windows(
    *,
    total_samples: int,
    max_chunk_samples: int,
) -> list[AudioChunk]:
    return [
        AudioChunk(
            start_sample=start,
            end_sample=min(start + max_chunk_samples, total_samples),
        )
        for start in range(0, total_samples, max_chunk_samples)
    ]


def _merge_adjacent_segments(
    segments: list[AudioChunk],
    *,
    max_chunk_samples: int,
    sample_rate: int,
    speech_pad_ms: int,
) -> list[AudioChunk]:
    if not segments:
        return []

    merged_segments: list[AudioChunk] = []
    edge_padding = int((speech_pad_ms / 1000) * sample_rate)

    current_start = segments[0].start_sample
    current_logical_start = segments[0].logical_start
    current_end = 0
    current_logical_end = current_logical_start

    for index, segment in enumerate(segments):
        start_sample = segment.start_sample
        end_sample = segment.end_sample

        if index > 0 and start_sample < segments[index - 1].end_sample:
            start_sample += edge_padding
        if index < len(segments) - 1 and end_sample > segments[index + 1].start_sample:
            end_sample -= edge_padding

        start_sample = max(0, start_sample)
        end_sample = max(start_sample + 1, end_sample)

        if end_sample - current_start > max_chunk_samples and current_end - current_start > 0:
            merged_segments.append(
                AudioChunk(
                    start_sample=current_start,
                    end_sample=current_end,
                    logical_start_sample=current_logical_start,
                    logical_end_sample=current_logical_end,
                )
            )
            current_start = start_sample
            current_logical_start = segment.logical_start

        current_end = end_sample
        current_logical_end = segment.logical_end

    merged_segments.append(
        AudioChunk(
            start_sample=current_start,
            end_sample=current_end,
            logical_start_sample=current_logical_start,
            logical_end_sample=current_logical_end,
        )
    )
    return merged_segments


def _get_text_separator(previous_text: str, current_text: str) -> str:
    previous_char = _last_non_space_char(previous_text)
    current_char = _first_non_space_char(current_text)

    if previous_char is None or current_char is None:
        return ""
    if current_char in NO_SPACE_BEFORE_CHARS:
        return ""
    if previous_char in NO_SPACE_AFTER_CHARS:
        return ""
    if previous_char in SENTENCE_END_CHARS:
        return " "
    if current_char.isalpha() and current_char.isupper():
        return ". "

    return " "


def _first_non_space_char(text: str) -> str | None:
    for char in text:
        if not char.isspace():
            return char
    return None


def _last_non_space_char(text: str) -> str | None:
    for char in reversed(text):
        if not char.isspace():
            return char
    return None
