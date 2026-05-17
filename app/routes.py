"""API routes for Omnilingual-ASR server."""

import logging
from typing import Literal

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from app.chunking import ChunkingConfig
from app.config import MODEL_NAME, ZERO_SHOT_MODEL_NAME
from app.exceptions import APIError
from app.handlers import handle_runtime_error

# Snapshot env defaults at import time. Used to show real defaults in the
# Swagger UI form via `example=` and `description=`. The actual fallback
# logic stays in `ChunkingConfig.from_env().with_overrides(...)` so leaving
# a field empty keeps env-driven defaults.
_DEFAULT_CHUNKING = ChunkingConfig.from_env()
from app.schemas import (
    ModelsResponse,
    TranscriptionResponse,
    TranscriptionSegmentResponse,
    TranscriptionWordResponse,
    VerboseTranscriptionResponse,
)
from app.service import DetailedTranscription, UploadedContextExample, asr_service

logger = logging.getLogger(__name__)

router = APIRouter()


def _configured_model_cards() -> list[dict[str, str | int]]:
    """Return API model entries for all configured model cards."""

    model_names = [MODEL_NAME]
    if ZERO_SHOT_MODEL_NAME and ZERO_SHOT_MODEL_NAME not in model_names:
        model_names.append(ZERO_SHOT_MODEL_NAME)

    return [
        {
            "id": model_name,
            "object": "model",
            "created": 0,
            "owned_by": "omnilingual-asr",
        }
        for model_name in model_names
    ]


async def _read_audio_upload(file: UploadFile) -> bytes:
    """Read and validate the primary upload."""

    if not file.filename:
        logger.warning("Transcription request rejected: no file provided")
        raise APIError(
            status_code=400,
            message="No file provided",
            param="file",
        )

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        logger.warning("Transcription request rejected: empty file")
        raise APIError(
            status_code=400,
            message="Empty file provided",
            param="file",
        )

    return audio_bytes


async def _read_context_examples(
    context_files: list[UploadFile] | None,
    context_texts: list[str] | None,
) -> list[UploadedContextExample]:
    """Read and validate zero-shot context uploads."""

    if not context_files or not context_texts:
        raise APIError(
            status_code=400,
            message=(
                "Zero-shot context requires both `context_files` and "
                "`context_texts`."
            ),
            param="context_files",
            code="missing_context_examples",
        )

    if len(context_files) != len(context_texts):
        raise APIError(
            status_code=400,
            message=(
                "`context_files` and `context_texts` must contain the same "
                "number of items."
            ),
            param="context_files",
            code="invalid_context_examples",
        )

    uploaded_context_examples: list[UploadedContextExample] = []
    for context_file, context_text in zip(context_files, context_texts, strict=True):
        context_bytes = await context_file.read()
        if len(context_bytes) == 0:
            raise APIError(
                status_code=400,
                message="Empty context file provided",
                param="context_files",
                code="invalid_context_examples",
            )

        uploaded_context_examples.append(
            UploadedContextExample(
                audio_bytes=context_bytes,
                text=context_text,
                content_type=context_file.content_type,
                filename=context_file.filename,
            )
        )

    return uploaded_context_examples


def _normalize_timestamp_granularities(
    timestamp_granularities: str | list[str] | None,
) -> set[Literal["word", "segment"]]:
    """Normalize flexible timestamp granularity inputs from multipart forms."""

    if timestamp_granularities is None:
        return set()

    normalized_values: set[Literal["word", "segment"]] = set()
    separators = [",", ";"]
    items = (
        [timestamp_granularities]
        if isinstance(timestamp_granularities, str)
        else list(timestamp_granularities)
    )
    for separator in separators:
        split_items: list[str] = []
        for item in items:
            split_items.extend(item.split(separator))
        items = split_items

    for raw_item in items:
        item = raw_item.strip().strip("[](){}").strip("'\"").lower()
        if not item:
            continue

        if item in {"word", "words", "word_timestamps"}:
            normalized_values.add("word")
        elif item in {"segment", "segments", "segment_timestamps"}:
            normalized_values.add("segment")

    return normalized_values


def _needs_detailed_transcription(
    response_format: str,
    timestamp_granularities: set[Literal["word", "segment"]],
) -> bool:
    """Return whether the request needs timing-aware transcription metadata."""

    return response_format in {"verbose_json", "srt", "vtt"} or bool(
        timestamp_granularities
    )


def _validate_timestamp_request(
    response_format: str,
    timestamp_granularities: set[Literal["word", "segment"]],
) -> None:
    """Reject timestamp requests that cannot be represented by the response format."""

    if timestamp_granularities and response_format != "verbose_json":
        raise APIError(
            status_code=400,
            message=(
                "`timestamp_granularities` requires "
                "`response_format=verbose_json`."
            ),
            param="response_format",
            code="invalid_timestamp_granularities",
        )


def _format_srt_timestamp(seconds: float) -> str:
    total_milliseconds = round(seconds * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def _format_vtt_timestamp(seconds: float) -> str:
    total_milliseconds = round(seconds * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:03}"


def _render_subtitle_response(
    result: DetailedTranscription,
    *,
    response_format: Literal["srt", "vtt"],
) -> PlainTextResponse:
    """Render VAD chunk segments as SRT or VTT subtitles."""

    if response_format == "vtt":
        lines = ["WEBVTT", ""]
        for segment in result.segments:
            lines.append(
                f"{_format_vtt_timestamp(segment.start_seconds)} --> "
                f"{_format_vtt_timestamp(segment.end_seconds)}"
            )
            lines.append(segment.text)
            lines.append("")
        return PlainTextResponse(content="\n".join(lines), media_type="text/vtt")

    lines: list[str] = []
    for index, segment in enumerate(result.segments, start=1):
        lines.append(str(index))
        lines.append(
            f"{_format_srt_timestamp(segment.start_seconds)} --> "
            f"{_format_srt_timestamp(segment.end_seconds)}"
        )
        lines.append(segment.text)
        lines.append("")

    return PlainTextResponse(content="\n".join(lines), media_type="text/plain")


def _format_transcription_response(
    result: str | DetailedTranscription,
    response_format: str,
    *,
    timestamp_granularities: set[Literal["word", "segment"]] | None = None,
):
    """Format a transcription response according to OpenAI-compatible rules."""

    if isinstance(result, str):
        text = result
    else:
        text = result.text

    if response_format == "text":
        return PlainTextResponse(content=text)

    if response_format in {"srt", "vtt"}:
        if isinstance(result, str):
            raise APIError(
                status_code=500,
                message="Detailed transcription is required for subtitle output.",
                error_type="server_error",
                code="missing_timestamps",
            )
        return _render_subtitle_response(result, response_format=response_format)

    if response_format == "verbose_json":
        if isinstance(result, str):
            raise APIError(
                status_code=500,
                message="Detailed transcription is required for verbose JSON output.",
                error_type="server_error",
                code="missing_timestamps",
            )

        include_words = timestamp_granularities is None or "word" in timestamp_granularities
        segments = [
            TranscriptionSegmentResponse(
                id=index,
                start=segment.start_seconds,
                end=segment.end_seconds,
                text=segment.text,
                language=segment.language,
                words=(
                    [
                        TranscriptionWordResponse(
                            word=word.word,
                            start=word.start_seconds,
                            end=word.end_seconds,
                            language=word.language,
                        )
                        for word in segment.words
                    ]
                    if include_words
                    else None
                ),
            )
            for index, segment in enumerate(result.segments)
        ]

        words = (
            [
                TranscriptionWordResponse(
                    word=word.word,
                    start=word.start_seconds,
                    end=word.end_seconds,
                    language=word.language,
                )
                for word in result.words
            ]
            if include_words
            else None
        )

        payload = VerboseTranscriptionResponse(
            language=result.language,
            duration=result.duration_seconds,
            text=result.text,
            words=words,
            segments=segments,
        )
        return JSONResponse(content=payload.model_dump())

    return JSONResponse(content=TranscriptionResponse(text=text).model_dump())


@router.get("/health-check")
@router.get("/healthz")
async def health_check():
    """Liveness probe — the process is up and serving HTTP."""
    return "ok"


@router.get("/readyz")
async def readiness_check():
    """Readiness probe — models are loaded on the device and inference is possible.

    Returns 503 until startup finishes. Used by load balancers and k8s to
    decide whether to route traffic to this replica.
    """
    if asr_service.standard_pipeline is None or asr_service.vad_model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "message": "Models are still initializing"},
        )
    return {"status": "ready"}


@router.get("/v1/models", response_model=ModelsResponse)
async def get_models():
    """Get model information."""
    return {"data": _configured_model_cards()}


@router.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(..., description="Primary audio file to transcribe."),
    model: str = Form(default=MODEL_NAME),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    timestamp_granularities: list[str] | None = Form(default=None),
    vad_threshold: float | None = Form(
        default=None,
        examples=[_DEFAULT_CHUNKING.threshold],
        description=(
            f"Silero VAD speech-probability threshold (0..1). "
            f"Default from env: {_DEFAULT_CHUNKING.threshold}. Higher = stricter."
        ),
    ),
    vad_neg_threshold: float | None = Form(
        default=None,
        examples=[_DEFAULT_CHUNKING.neg_threshold],
        description=(
            "Optional VAD hysteresis lower threshold (0..1). Leave empty to "
            f"use Silero default. Current env value: "
            f"{_DEFAULT_CHUNKING.neg_threshold!r}."
        ),
    ),
    vad_min_speech_ms: int | None = Form(
        default=None,
        examples=[_DEFAULT_CHUNKING.min_speech_ms],
        description=(
            f"Minimum speech segment in ms; shorter is dropped. "
            f"Default from env: {_DEFAULT_CHUNKING.min_speech_ms}."
        ),
    ),
    vad_min_silence_ms: int | None = Form(
        default=None,
        examples=[_DEFAULT_CHUNKING.min_silence_ms],
        description=(
            f"Silence in ms that triggers a chunk break. Lower = chunks split more often. "
            f"Default from env: {_DEFAULT_CHUNKING.min_silence_ms}."
        ),
    ),
    vad_speech_pad_ms: int | None = Form(
        default=None,
        examples=[_DEFAULT_CHUNKING.speech_pad_ms],
        description=(
            f"Padding in ms added to each VAD chunk for model context. "
            f"Default from env: {_DEFAULT_CHUNKING.speech_pad_ms}."
        ),
    ),
    chunk_max_seconds: float | None = Form(
        default=None,
        examples=[_DEFAULT_CHUNKING.chunk_max_seconds],
        description=(
            f"Max chunk length in seconds (model cap is 40). "
            f"Default from env: {_DEFAULT_CHUNKING.chunk_max_seconds}."
        ),
    ),
):
    """
    OpenAI Whisper-compatible transcription endpoint for the standard model.

    Args:
        file: Audio file (wav, mp3, flac, etc.)
        model: Model identifier (informational only)
        language: Language code (ISO 639-1 or Omnilingual-ASR format)
        prompt: Optional prompt (not used)
        response_format: json, verbose_json, text, srt, or vtt
        temperature: Sampling temperature (not used)
        timestamp_granularities: Timestamp detail level for verbose JSON output
    """
    del model, prompt, temperature

    audio_bytes = await _read_audio_upload(file)
    normalized_timestamp_granularities = _normalize_timestamp_granularities(
        timestamp_granularities
    )
    _validate_timestamp_request(response_format, normalized_timestamp_granularities)
    needs_detail = _needs_detailed_transcription(
        response_format,
        normalized_timestamp_granularities,
    )
    chunking_config = ChunkingConfig.from_env().with_overrides(
        threshold=vad_threshold,
        neg_threshold=vad_neg_threshold,
        min_speech_ms=vad_min_speech_ms,
        min_silence_ms=vad_min_silence_ms,
        speech_pad_ms=vad_speech_pad_ms,
        chunk_max_seconds=chunk_max_seconds,
    )

    logger.info(
        "Standard transcription request: file=%s, language=%s, format=%s, timestamps=%s",
        file.filename,
        language,
        response_format,
        sorted(normalized_timestamp_granularities),
    )

    try:
        if needs_detail:
            result = await asr_service.transcribe_detailed(
                audio_bytes,
                language=language,
                content_type=file.content_type,
                filename=file.filename,
                chunking_config=chunking_config,
            )
        else:
            result = await asr_service.transcribe(
                audio_bytes,
                language=language,
                content_type=file.content_type,
                filename=file.filename,
                chunking_config=chunking_config,
            )
    except RuntimeError as e:
        logger.exception("Standard transcription failed for %s", file.filename)
        handle_runtime_error(e)
    except Exception as e:
        logger.exception("Standard transcription failed for %s", file.filename)
        raise APIError(
            status_code=500,
            message=f"Transcription failed: {e}",
            error_type="server_error",
        )

    return _format_transcription_response(
        result,
        response_format,
        timestamp_granularities=normalized_timestamp_granularities,
    )


@router.post("/v1/audio/transcriptions/zero-shot")
async def transcribe_zero_shot(
    file: UploadFile = File(..., description="Primary audio file to transcribe."),
    model: str | None = Form(default=ZERO_SHOT_MODEL_NAME),
    response_format: str = Form(default="json"),
    prompt: str | None = Form(default=None),
    temperature: float = Form(default=0.0),
    timestamp_granularities: list[str] | None = Form(default=None),
    context_files: list[UploadFile] | None = File(
        default=None,
        description=(
            "Zero-shot context audio examples. Provide one or more files in the same "
            "order as `context_texts`."
        ),
    ),
    context_texts: list[str] | None = Form(
        default=None,
        description=(
            "Zero-shot transcripts aligned with `context_files`. Each item must be "
            "the transcript of the file at the same index."
        ),
    ),
    vad_threshold: float | None = Form(
        default=None,
        examples=[_DEFAULT_CHUNKING.threshold],
        description=f"Default from env: {_DEFAULT_CHUNKING.threshold}.",
    ),
    vad_neg_threshold: float | None = Form(
        default=None,
        examples=[_DEFAULT_CHUNKING.neg_threshold],
        description=f"Default from env: {_DEFAULT_CHUNKING.neg_threshold!r}.",
    ),
    vad_min_speech_ms: int | None = Form(
        default=None,
        examples=[_DEFAULT_CHUNKING.min_speech_ms],
        description=f"Default from env: {_DEFAULT_CHUNKING.min_speech_ms}.",
    ),
    vad_min_silence_ms: int | None = Form(
        default=None,
        examples=[_DEFAULT_CHUNKING.min_silence_ms],
        description=f"Default from env: {_DEFAULT_CHUNKING.min_silence_ms}.",
    ),
    vad_speech_pad_ms: int | None = Form(
        default=None,
        examples=[_DEFAULT_CHUNKING.speech_pad_ms],
        description=f"Default from env: {_DEFAULT_CHUNKING.speech_pad_ms}.",
    ),
    chunk_max_seconds: float | None = Form(
        default=None,
        examples=[_DEFAULT_CHUNKING.chunk_max_seconds],
        description=f"Default from env: {_DEFAULT_CHUNKING.chunk_max_seconds}.",
    ),
):
    """
    Zero-shot transcription endpoint using a dedicated `*_ZS` pipeline.

    Args:
        file: Audio file (wav, mp3, flac, etc.)
        model: Model identifier (informational only)
        response_format: json, verbose_json, text, srt, or vtt
        prompt: Optional prompt (not used)
        temperature: Sampling temperature (not used)
        timestamp_granularities: Timestamp detail level for verbose JSON output
        context_files: One or more aligned context audio files
        context_texts: One or more aligned context transcripts
    """
    del model, prompt, temperature

    audio_bytes = await _read_audio_upload(file)
    uploaded_context_examples = await _read_context_examples(
        context_files,
        context_texts,
    )
    normalized_timestamp_granularities = _normalize_timestamp_granularities(
        timestamp_granularities
    )
    _validate_timestamp_request(response_format, normalized_timestamp_granularities)
    needs_detail = _needs_detailed_transcription(
        response_format,
        normalized_timestamp_granularities,
    )
    chunking_config = ChunkingConfig.from_env().with_overrides(
        threshold=vad_threshold,
        neg_threshold=vad_neg_threshold,
        min_speech_ms=vad_min_speech_ms,
        min_silence_ms=vad_min_silence_ms,
        speech_pad_ms=vad_speech_pad_ms,
        chunk_max_seconds=chunk_max_seconds,
    )

    logger.info(
        "Zero-shot transcription request: file=%s, contexts=%s, format=%s, timestamps=%s",
        file.filename,
        len(uploaded_context_examples),
        response_format,
        sorted(normalized_timestamp_granularities),
    )

    try:
        if needs_detail:
            result = await asr_service.transcribe_zero_shot_detailed(
                audio_bytes,
                content_type=file.content_type,
                filename=file.filename,
                context_examples=uploaded_context_examples,
                chunking_config=chunking_config,
            )
        else:
            result = await asr_service.transcribe_zero_shot(
                audio_bytes,
                content_type=file.content_type,
                filename=file.filename,
                context_examples=uploaded_context_examples,
                chunking_config=chunking_config,
            )
    except RuntimeError as e:
        logger.exception("Zero-shot transcription failed for %s", file.filename)
        handle_runtime_error(e)
    except Exception as e:
        logger.exception("Zero-shot transcription failed for %s", file.filename)
        raise APIError(
            status_code=500,
            message=f"Transcription failed: {e}",
            error_type="server_error",
        )

    return _format_transcription_response(
        result,
        response_format,
        timestamp_granularities=normalized_timestamp_granularities,
    )
