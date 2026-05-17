"""
OpenAI Whisper-compatible response schemas for Omnilingual-ASR.
"""

from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    """Standard transcription response (response_format=json).

    See: https://platform.openai.com/docs/api-reference/audio/json-object
    """

    text: str = Field(..., description="The transcribed text")


class TranscriptionWordResponse(BaseModel):
    """Approximate word-level timestamp information."""

    word: str = Field(..., description="The recognized word token")
    start: float = Field(..., description="Approximate word start time in seconds")
    end: float = Field(..., description="Approximate word end time in seconds")
    language: str | None = Field(
        None,
        description="Detected language of the parent chunk (if LID enabled)",
    )


class TranscriptionSegmentResponse(BaseModel):
    """Speech-aware transcript segment with timing metadata."""

    id: int = Field(..., description="Zero-based segment index")
    start: float = Field(..., description="Segment start time in seconds")
    end: float = Field(..., description="Segment end time in seconds")
    text: str = Field(..., description="Recognized text for the segment")
    language: str | None = Field(
        None,
        description="Detected language of this segment (if LID enabled)",
    )
    words: list[TranscriptionWordResponse] | None = Field(
        None,
        description="Approximate word timestamps within this segment",
    )


class VerboseTranscriptionResponse(BaseModel):
    """Verbose transcription response similar to Whisper/OpenAI verbose_json."""

    task: str = Field("transcribe", description="The executed task")
    language: str | None = Field(None, description="Language hint used for decoding")
    duration: float = Field(..., description="Decoded audio duration in seconds")
    text: str = Field(..., description="The stitched transcript")
    words: list[TranscriptionWordResponse] | None = Field(
        None,
        description="Approximate word timestamps across the full transcript",
    )
    segments: list[TranscriptionSegmentResponse] = Field(
        ...,
        description="Speech-aware transcript segments",
    )


class ErrorResponse(BaseModel):
    """Error response format matching OpenAI's error schema."""

    class ErrorInfo(BaseModel):
        message: str = Field(..., description="The error message")
        type: str = Field(..., description="The error type")
        param: str | None = Field(None, description="The error parameter")
        code: str | None = Field(None, description="The error code")

    error: ErrorInfo = Field(..., description="Error details")


class ModelsResponse(BaseModel):
    """Models response format matching OpenAI's models schema.

    See: https://platform.openai.com/docs/api-reference/models/list
    """

    class ModelInfo(BaseModel):
        id: str = Field(..., description="The model identifier")
        object: str = Field(..., description="The object type")
        created: int = Field(0, description="The creation timestamp")
        owned_by: str = Field(..., description="The owner of the model")

    data: list[ModelInfo] = Field(..., description="List of model information")
