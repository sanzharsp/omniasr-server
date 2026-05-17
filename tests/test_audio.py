"""Tests for audio decoding and chunk planning helpers."""

import asyncio
import io
import math
import struct
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from app.audio import (
    AudioChunk,
    DecodedAudio,
    build_timed_segments,
    build_speech_chunks,
    decode_audio_bytes,
    join_transcript_texts,
)
from app.chunking import ChunkingConfig
from app.service import DetailedTranscription, OmnilingualASRService, UploadedContextExample


def _vad_config(
    *,
    threshold: float = 0.5,
    neg_threshold: float | None = None,
    min_speech_ms: int = 250,
    min_silence_ms: int = 300,
    speech_pad_ms: int = 200,
    chunk_max_seconds: float = 30.0,
) -> ChunkingConfig:
    return ChunkingConfig(
        threshold=threshold,
        neg_threshold=neg_threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        chunk_max_seconds=chunk_max_seconds,
    )


def _build_wave_bytes(
    *,
    duration_seconds: float,
    sample_rate: int = 8000,
    channels: int = 1,
) -> bytes:
    """Create an in-memory PCM WAV clip for decode tests."""

    sample_count = int(duration_seconds * sample_rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        for index in range(sample_count):
            sample = int(32767 * math.sin(2 * math.pi * 440 * (index / sample_rate)))
            frame = struct.pack("<h", sample)
            wav_file.writeframes(frame * channels)

    return buffer.getvalue()


def test_decode_audio_bytes_resamples_and_converts_to_mono():
    """Decoded audio should be normalized to mono 16kHz float32."""

    decoded = decode_audio_bytes(
        _build_wave_bytes(duration_seconds=1.0, channels=2),
        filename="sample.wav",
        content_type="audio/wav",
    )

    assert decoded.sample_rate == 16000
    assert decoded.waveform.ndim == 1
    assert decoded.waveform.dtype == np.float32
    assert len(decoded.waveform) == 16000


def test_chunking_config_with_overrides_replaces_only_provided_fields():
    """Overrides should be a copy-on-write — None means keep, value means replace."""

    base = _vad_config(threshold=0.5, min_speech_ms=250, speech_pad_ms=200)

    updated = base.with_overrides(threshold=0.7, speech_pad_ms=100)

    assert updated.threshold == 0.7
    assert updated.speech_pad_ms == 100
    assert updated.min_speech_ms == 250  # unchanged
    assert updated.chunk_max_seconds == base.chunk_max_seconds
    assert base.threshold == 0.5  # original untouched


def test_decode_audio_bytes_supports_raw_pcm():
    """Raw s16le bytes should be accepted for audio/raw uploads."""

    pcm = (np.array([0, 32767, -32768], dtype=np.int16)).tobytes()

    decoded = decode_audio_bytes(
        pcm,
        content_type="audio/raw",
        filename="sample.pcm",
    )

    assert decoded.sample_rate == 16000
    assert decoded.waveform.dtype == np.float32
    assert decoded.waveform.shape == (3,)


def test_build_speech_chunks_falls_back_to_fixed_sized_windows_for_silence():
    """Long silence should still be chunked safely below the model limit."""

    audio = DecodedAudio(
        waveform=np.zeros(65 * 16000, dtype=np.float32),
        sample_rate=16000,
    )

    with patch("app.audio.get_speech_timestamps", return_value=[]):
        chunks = build_speech_chunks(
            audio,
            vad_model=object(),
            vad_device=torch.device("cpu"),
            config=_vad_config(),
        )

    assert len(chunks) == 3
    assert [chunk.end_sample - chunk.start_sample for chunk in chunks] == [
        30 * 16000,
        30 * 16000,
        5 * 16000,
    ]


def test_build_speech_chunks_merges_silero_segments_within_limit():
    """Adjacent Silero segments should be merged into larger batched chunks."""

    audio = DecodedAudio(
        waveform=np.zeros(50 * 16000, dtype=np.float32),
        sample_rate=16000,
    )
    speech_segments = [
        {"start": 0, "end": 10 * 16000},
        {"start": 12 * 16000, "end": 22 * 16000},
        {"start": 24 * 16000, "end": 34 * 16000},
    ]

    with patch("app.audio.get_speech_timestamps", return_value=speech_segments):
        chunks = build_speech_chunks(
            audio,
            vad_model=object(),
            vad_device=torch.device("cpu"),
            config=_vad_config(),
        )

    assert [(chunk.start_sample, chunk.end_sample) for chunk in chunks] == [
        (0, 22 * 16000),
        (24 * 16000, 34 * 16000),
    ]


def test_join_transcript_texts_normalizes_spacing():
    """Chunked transcription text should stitch back together cleanly."""

    result = join_transcript_texts(["Hello", "world", "!", "How are you?"])

    assert result == "Hello world! How are you?"


def test_join_transcript_texts_inserts_sentence_break_for_new_capitalized_chunk():
    """A capitalized chunk after plain text should start a new sentence."""

    result = join_transcript_texts(["hello there", "General Kenobi"])

    assert result == "hello there. General Kenobi"


def test_build_timed_segments_distributes_chunk_time_across_words():
    """Timed segments should preserve chunk times and emit approximate words."""

    segments = build_timed_segments(
        [AudioChunk(0, 32000)],
        ["hello wide world"],
        sample_rate=16000,
    )

    assert len(segments) == 1
    assert segments[0].start_seconds == 0.0
    assert segments[0].end_seconds == 2.0
    assert [word.word for word in segments[0].words] == ["hello", "wide", "world"]
    assert segments[0].words[0].start_seconds == 0.0
    assert segments[0].words[-1].end_seconds == 2.0


def test_build_timed_segments_uses_logical_bounds_not_padded():
    """Segment and word times must come from un-padded VAD bounds, not the
    padded inference window. Without this the response would drift by
    speech_pad_ms on every chunk."""

    chunk = AudioChunk(
        start_sample=0,
        end_sample=32000,
        logical_start_sample=3200,  # +200ms inward from padded start
        logical_end_sample=28800,  # -200ms inward from padded end
    )

    segments = build_timed_segments([chunk], ["hello world"], sample_rate=16000)

    assert segments[0].start_seconds == 0.2
    assert segments[0].end_seconds == 1.8
    assert segments[0].words[0].start_seconds == 0.2
    assert segments[0].words[-1].end_seconds == 1.8


def test_build_speech_chunks_recovers_logical_bounds_from_silero_pad():
    """Silero inflates segments by speech_pad_ms; logical bounds must shrink
    them back so downstream timestamps are not delayed by pad_ms."""

    audio = DecodedAudio(
        waveform=np.zeros(10 * 16000, dtype=np.float32),
        sample_rate=16000,
    )
    # Silero returns padded segment: VAD detected [24000, 56000], padding
    # widens it to [20800, 59200] (pad_ms=200 -> 3200 samples).
    speech_segments = [{"start": 20800, "end": 59200}]

    with patch("app.audio.get_speech_timestamps", return_value=speech_segments):
        chunks = build_speech_chunks(
            audio,
            vad_model=object(),
            vad_device=torch.device("cpu"),
            config=_vad_config(),
        )

    assert len(chunks) == 1
    assert chunks[0].start_sample == 20800
    assert chunks[0].end_sample == 59200
    assert chunks[0].logical_start == 24000
    assert chunks[0].logical_end == 56000


def test_service_transcribe_batches_chunked_inputs():
    """Service should decode once, transcribe chunks in a batch, and stitch text."""

    service = OmnilingualASRService()
    service.standard_model_name = "omniASR_LLM_1B_v2"
    service.standard_pipeline = MagicMock()
    service.vad_model = object()
    service.standard_pipeline.transcribe.return_value = ["Hello", "world", "!"]

    decoded_audio = DecodedAudio(
        waveform=np.zeros(3 * 16000, dtype=np.float32),
        sample_rate=16000,
    )
    chunks = [
        AudioChunk(0, 16000),
        AudioChunk(16000, 32000),
        AudioChunk(32000, 48000),
    ]

    with (
        patch("app.service.OMNILINGUAL_BATCH_SIZE", 8),
        patch("app.service.decode_audio_bytes", return_value=decoded_audio),
        patch("app.service.build_speech_chunks", return_value=chunks),
    ):
        result = asyncio.run(service.transcribe(b"fake-audio", language="en"))

    assert result == "Hello world!"

    service.standard_pipeline.transcribe.assert_called_once()
    call_args, call_kwargs = service.standard_pipeline.transcribe.call_args
    assert len(call_args[0]) == 3
    assert call_kwargs["batch_size"] == 3
    assert call_kwargs["lang"] == ["eng_Latn", "eng_Latn", "eng_Latn"]


def test_service_transcribe_detailed_returns_segments_and_words():
    """Detailed transcription should keep VAD chunk timings in the result."""

    service = OmnilingualASRService()
    service.standard_model_name = "omniASR_LLM_1B_v2"
    service.standard_pipeline = MagicMock()
    service.vad_model = object()
    service.standard_pipeline.transcribe.return_value = ["hello there", "General Kenobi"]

    decoded_audio = DecodedAudio(
        waveform=np.zeros(3 * 16000, dtype=np.float32),
        sample_rate=16000,
    )
    chunks = [
        AudioChunk(0, 16000),
        AudioChunk(16000, 48000),
    ]

    with (
        patch("app.service.OMNILINGUAL_BATCH_SIZE", 8),
        patch("app.service.decode_audio_bytes", return_value=decoded_audio),
        patch("app.service.build_speech_chunks", return_value=chunks),
    ):
        result = asyncio.run(service.transcribe_detailed(b"fake-audio", language="en"))

    assert isinstance(result, DetailedTranscription)
    assert result.text == "hello there. General Kenobi"
    assert len(result.segments) == 2
    assert result.segments[0].start_seconds == 0.0
    assert result.segments[0].end_seconds == 1.0
    assert result.segments[1].start_seconds == 1.0
    assert result.segments[1].end_seconds == 3.0
    assert [word.word for word in result.words] == [
        "hello",
        "there",
        "General",
        "Kenobi",
    ]


def test_decode_context_examples_splits_long_audio_into_fixed_windows():
    """Long zero-shot context audio should be expanded into <=40s windows."""

    service = OmnilingualASRService()
    long_context_audio = DecodedAudio(
        waveform=np.zeros(85 * 16000, dtype=np.float32),
        sample_rate=16000,
    )

    with patch("app.service.decode_audio_bytes", return_value=long_context_audio):
        decoded_examples = service._decode_context_examples(
            [
                UploadedContextExample(
                    audio_bytes=b"long-context",
                    text="one two three four five six seven eight nine ten eleven twelve",
                    content_type="audio/wav",
                    filename="ctx.wav",
                )
            ]
        )

    assert len(decoded_examples) == 3
    assert decoded_examples[0].audio["waveform"].shape[0] == 40 * 16000
    assert decoded_examples[1].audio["waveform"].shape[0] == 40 * 16000
    assert decoded_examples[2].audio["waveform"].shape[0] == 5 * 16000
    assert all(example.text for example in decoded_examples)
    assert " ".join(example.text for example in decoded_examples).split() == [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
    ]


def test_decode_context_examples_repeats_short_transcript_when_split_needed():
    """Very short transcripts should be repeated when a safe split is impossible."""

    service = OmnilingualASRService()
    long_context_audio = DecodedAudio(
        waveform=np.zeros(85 * 16000, dtype=np.float32),
        sample_rate=16000,
    )

    with patch("app.service.decode_audio_bytes", return_value=long_context_audio):
        decoded_examples = service._decode_context_examples(
            [
                UploadedContextExample(
                    audio_bytes=b"long-context",
                    text="hi",
                    content_type="audio/wav",
                    filename="ctx.wav",
                )
            ]
        )

    assert len(decoded_examples) == 3
    assert [example.text for example in decoded_examples] == ["hi", "hi", "hi"]


def test_service_transcribe_zero_shot_uses_context_for_each_chunk():
    """Zero-shot transcription should call transcribe_with_context per chunk."""

    service = OmnilingualASRService()
    service.zero_shot_model_name = "omniASR_LLM_7B_ZS"
    service.zero_shot_pipeline = MagicMock()
    service.vad_model = object()
    service.zero_shot_pipeline.transcribe_with_context.return_value = ["Hello", "world"]

    decoded_audio = DecodedAudio(
        waveform=np.zeros(2 * 16000, dtype=np.float32),
        sample_rate=16000,
    )
    context_audio = DecodedAudio(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
    )
    chunks = [
        AudioChunk(0, 16000),
        AudioChunk(16000, 32000),
    ]

    with (
        patch("app.service.OMNILINGUAL_BATCH_SIZE", 8),
        patch("app.service.decode_audio_bytes", side_effect=[decoded_audio, context_audio]),
        patch("app.service.build_speech_chunks", return_value=chunks),
    ):
        result = asyncio.run(
            service.transcribe_zero_shot(
                b"fake-audio",
                context_examples=[
                    UploadedContextExample(
                        audio_bytes=b"context-audio",
                        text="hello world",
                        content_type="audio/wav",
                        filename="ctx.wav",
                    )
                ],
            )
        )

    assert result == "Hello world"

    service.zero_shot_pipeline.transcribe_with_context.assert_called_once()
    call_args, call_kwargs = service.zero_shot_pipeline.transcribe_with_context.call_args
    assert len(call_args[0]) == 2
    assert call_kwargs["batch_size"] == 1
    assert len(call_kwargs["context_examples"]) == 2
    assert len(call_kwargs["context_examples"][0]) == 1
    assert call_kwargs["context_examples"][0][0].text == "hello world"
