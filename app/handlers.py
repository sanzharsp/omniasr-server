"""
Exception handlers for OpenAI-compatible error responses.
"""

import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.exceptions import APIError
from app.schemas import ErrorResponse

logger = logging.getLogger(__name__)


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle APIError exceptions with OpenAI-compatible error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorResponse.ErrorInfo(
                message=exc.message,
                type=exc.error_type,
                param=exc.param,
                code=exc.code,
            )
        ).model_dump(),
    )


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle validation errors with OpenAI-compatible error responses."""
    errors = exc.errors()
    if errors:
        first_error = errors[0]
        loc = first_error.get("loc", ())
        param = ".".join(str(x) for x in loc) if loc else None
        message = first_error.get("msg", "Validation error")
    else:
        param = None
        message = "Validation error"

    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=ErrorResponse.ErrorInfo(
                message=message,
                type="invalid_request_error",
                param=param,
            )
        ).model_dump(),
    )


def handle_runtime_error(e: RuntimeError) -> None:
    """Handle runtime errors with OpenAI-compatible error responses."""
    message = str(e).lower()
    cause = e.__cause__
    cause_msg = str(cause).lower() if cause else ""
    cause_name = cause.__class__.__name__.lower() if cause else ""

    if "zero-shot model is not configured" in message:
        raise APIError(
            status_code=503,
            message=(
                "Zero-shot transcription is not configured on this server. "
                "Set ZERO_SHOT_MODEL_NAME to enable the zero-shot endpoint."
            ),
            error_type="server_error",
            param="model",
            code="model_unavailable",
        )

    if "standard non-zero-shot model" in message or "must point to a *_zs model" in message:
        raise APIError(
            status_code=500,
            message="Server model configuration is invalid.",
            error_type="server_error",
            code="invalid_model_configuration",
        )

    if "requires context examples" in message or "context conditioning" in message:
        raise APIError(
            status_code=400,
            message=(
                "Zero-shot transcription requires matching `context_files` and "
                "`context_texts` form fields."
            ),
            param="context_files",
            code="missing_context_examples",
        )

    if "zero-shot transcription requires at least one context example" in message:
        raise APIError(
            status_code=400,
            message=(
                "Zero-shot transcription requires matching `context_files` and "
                "`context_texts` form fields."
            ),
            param="context_files",
            code="missing_context_examples",
        )

    if "context example" in message and "empty transcription" in message:
        raise APIError(
            status_code=400,
            message="Each context example must include a non-empty transcript.",
            param="context_texts",
            code="invalid_context_examples",
        )

    if "context example" in message and "exceeds the 40s model limit" in message:
        raise APIError(
            status_code=400,
            message="Each context example must be 40 seconds or shorter.",
            param="context_files",
            code="invalid_context_examples",
        )

    if "context audio decode failed" in message:
        raise APIError(
            status_code=400,
            message="Could not decode one of the context audio files.",
            param="context_files",
            code="invalid_audio_format",
        )

    if (
        "sndfile" in cause_msg
        or "decode" in cause_msg
        or "invaliddataerror" in cause_name
        or "ffmpegerror" in cause_name
        or "averror" in cause_name
    ):
        raise APIError(
            status_code=400,
            message="Could not decode audio file. The file may be corrupted or in an unsupported format.",
            param="file",
            code="invalid_audio_format",
        )

    if "max audio length" in cause_msg:
        raise APIError(
            status_code=500,
            message=(
                "Transcription chunk exceeded the underlying model limit. "
                "Check chunking configuration."
            ),
            error_type="server_error",
            code="invalid_chunk_length",
        )

    raise APIError(
        status_code=500,
        message=f"Transcription failed: {e}. Cause: {cause_msg or 'unknown'}",
        error_type="server_error",
    )
