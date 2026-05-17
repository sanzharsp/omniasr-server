"""Integration tests for API routes."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.server import app
from app.audio import TimedTranscriptSegment, WordTimestamp
from app.service import DetailedTranscription


@pytest.fixture
def client():
    """Create a test client with mocked ASR service."""

    with patch("app.service.asr_service") as mock_service:
        mock_service.load_model = MagicMock()
        with TestClient(app) as test_client:
            yield test_client


def test_get_health_check(client: TestClient):
    """Health check should return valid JSON."""

    response = client.get("/health-check")

    assert response.headers["content-type"] == "application/json"
    assert response.status_code == 200
    assert response.text == '"ok"'


def test_get_models(client: TestClient):
    """Models endpoint should return at least the standard model."""

    response = client.get("/v1/models")

    assert response.status_code == 200

    data = response.json()

    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) >= 1

    model = response.json()["data"][0]

    assert "id" in model
    assert "object" in model
    assert "created" in model
    assert "owned_by" in model
    assert model["object"] == "model"
    assert model["owned_by"] == "omnilingual-asr"
    assert isinstance(model["created"], int)


@patch("app.routes.ZERO_SHOT_MODEL_NAME", "omniASR_LLM_7B_ZS")
@patch("app.routes.MODEL_NAME", "omniASR_CTC_300M_v2")
def test_get_models_includes_zero_shot_model_when_configured(client: TestClient):
    """The models list should expose both configured model cards."""

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"] == [
        {
            "id": "omniASR_CTC_300M_v2",
            "object": "model",
            "created": 0,
            "owned_by": "omnilingual-asr",
        },
        {
            "id": "omniASR_LLM_7B_ZS",
            "object": "model",
            "created": 0,
            "owned_by": "omnilingual-asr",
        },
    ]


def test_app_uses_configured_root_path():
    """Prefixed routes should work for direct local access."""

    client = TestClient(app)

    assert client.get("/omni-asr/health-check").status_code == 200
    assert client.get("/omni-asr/docs").status_code == 200
    assert client.get("/omni-asr/openapi.json").status_code == 200


def test_transcribe_forwards_upload_metadata(client: TestClient):
    """The standard route should pass filename and content type through."""

    with patch("app.routes.asr_service.transcribe", return_value="hello") as mock_transcribe:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", b"RIFFfake", "audio/wav")},
        )

    assert response.status_code == 200
    mock_transcribe.assert_called_once()
    call_args, call_kwargs = mock_transcribe.call_args
    assert call_args == (b"RIFFfake",)
    assert call_kwargs["language"] is None
    assert call_kwargs["content_type"] == "audio/wav"
    assert call_kwargs["filename"] == "sample.wav"
    assert call_kwargs["chunking_config"] is not None


def test_transcribe_verbose_json_includes_segments_and_words(client: TestClient):
    """Verbose JSON should expose segment and approximate word timestamps."""

    detailed = DetailedTranscription(
        text="hello world",
        duration_seconds=2.0,
        language="eng_Latn",
        segments=[
            TimedTranscriptSegment(
                start_seconds=0.0,
                end_seconds=2.0,
                text="hello world",
                words=[
                    WordTimestamp("hello", 0.0, 1.0),
                    WordTimestamp("world", 1.0, 2.0),
                ],
            )
        ],
        words=[
            WordTimestamp("hello", 0.0, 1.0),
            WordTimestamp("world", 1.0, 2.0),
        ],
    )

    with patch("app.routes.asr_service.transcribe_detailed", return_value=detailed) as mock_transcribe:
        response = client.post(
            "/v1/audio/transcriptions",
            data={
                "response_format": "verbose_json",
                "timestamp_granularities": "word_timestamps",
            },
            files={"file": ("sample.wav", b"RIFFfake", "audio/wav")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "hello world"
    assert data["duration"] == 2.0
    assert data["segments"][0]["text"] == "hello world"
    assert data["segments"][0]["words"][0] == {
        "word": "hello",
        "start": 0.0,
        "end": 1.0,
        "language": None,
    }
    assert data["words"][1] == {
        "word": "world",
        "start": 1.0,
        "end": 2.0,
        "language": None,
    }
    mock_transcribe.assert_called_once()
    call_args, call_kwargs = mock_transcribe.call_args
    assert call_args == (b"RIFFfake",)
    assert call_kwargs["language"] is None
    assert call_kwargs["content_type"] == "audio/wav"
    assert call_kwargs["filename"] == "sample.wav"
    assert call_kwargs["chunking_config"] is not None


def test_transcribe_verbose_json_accepts_repeated_timestamp_granularities(
    client: TestClient,
):
    """Repeated multipart timestamp granularity fields should preserve both values."""

    detailed = DetailedTranscription(
        text="hello world",
        duration_seconds=2.0,
        language="eng_Latn",
        segments=[
            TimedTranscriptSegment(
                start_seconds=0.0,
                end_seconds=2.0,
                text="hello world",
                words=[
                    WordTimestamp("hello", 0.0, 1.0),
                    WordTimestamp("world", 1.0, 2.0),
                ],
            )
        ],
        words=[
            WordTimestamp("hello", 0.0, 1.0),
            WordTimestamp("world", 1.0, 2.0),
        ],
    )

    with patch("app.routes.asr_service.transcribe_detailed", return_value=detailed):
        response = client.post(
            "/v1/audio/transcriptions",
            files=[
                ("file", ("sample.wav", b"RIFFfake", "audio/wav")),
                ("response_format", (None, "verbose_json")),
                ("timestamp_granularities", (None, "word")),
                ("timestamp_granularities", (None, "segment")),
            ],
        )

    assert response.status_code == 200
    data = response.json()
    assert data["segments"][0]["start"] == 0.0
    assert data["segments"][0]["words"][0]["word"] == "hello"


def test_transcribe_rejects_timestamp_granularities_without_verbose_json(
    client: TestClient,
):
    """Timestamp granularity requests should fail fast on incompatible formats."""

    response = client.post(
        "/v1/audio/transcriptions",
        data={"timestamp_granularities": "word"},
        files={"file": ("sample.wav", b"RIFFfake", "audio/wav")},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "response_format"
    assert error["code"] == "invalid_timestamp_granularities"


def test_zero_shot_transcribe_forwards_context_examples(client: TestClient):
    """Zero-shot route should pair context files/texts and forward them."""

    with patch(
        "app.routes.asr_service.transcribe_zero_shot",
        return_value="hello",
    ) as mock_transcribe:
        response = client.post(
            "/v1/audio/transcriptions/zero-shot",
            files=[
                ("file", ("sample.wav", b"RIFFfake", "audio/wav")),
                ("context_files", ("ctx.wav", b"RIFFcontext", "audio/wav")),
                ("context_texts", (None, "first context transcript")),
            ],
        )

    assert response.status_code == 200
    forwarded_context_examples = mock_transcribe.call_args.kwargs["context_examples"]
    assert forwarded_context_examples is not None
    assert len(forwarded_context_examples) == 1
    assert forwarded_context_examples[0].text == "first context transcript"
    assert forwarded_context_examples[0].filename == "ctx.wav"


def test_zero_shot_transcribe_rejects_mismatched_context_examples(client: TestClient):
    """Context file/text arrays must have matching lengths."""

    response = client.post(
        "/v1/audio/transcriptions/zero-shot",
        files=[
            ("file", ("sample.wav", b"RIFFfake", "audio/wav")),
            ("context_files", ("ctx.wav", b"RIFFcontext", "audio/wav")),
            ("context_texts", (None, "first")),
            ("context_texts", (None, "second")),
        ],
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "context_files"
    assert error["code"] == "invalid_context_examples"


def test_zero_shot_transcribe_requires_context_examples(client: TestClient):
    """Zero-shot requests should fail fast when context is missing."""

    response = client.post(
        "/v1/audio/transcriptions/zero-shot",
        files={"file": ("sample.wav", b"RIFFfake", "audio/wav")},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "context_files"
    assert error["code"] == "missing_context_examples"
