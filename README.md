# Omnilingual-ASR Model Server

A FastAPI-based ASR model server for [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) with an OpenAI Whisper-compatible API.

## Quick Start

### Local Development

```bash
# Install dependencies with uv
uv sync

# Create local environment file
cp .env.example .env

# Run the server
uv run python main.py
```

## Building a Docker Image

This repo builds the image with CUDA 12.8 and PyTorch 2.8.0. To build against a different CUDA version, you need to update the sources and indices in [pyproject.toml](pyproject.toml).

### Helpful resources

- Supported combinations of CUDA, PyTorch, and Python for `fairseq2`: https://github.com/facebookresearch/fairseq2?tab=readme-ov-file#variants
- Organizing sources and indices: https://docs.astral.sh/uv/concepts/indexes/

### Using the build script

The [`build.sh`](build.sh) script is a good place to start your own builds:

```bash
# Build with default model
bash build.sh

# Build with a standard model plus a dedicated zero-shot model
MODEL_NAME=omniASR_CTC_300M_v2 ZERO_SHOT_MODEL_NAME=omniASR_LLM_7B_ZS bash build.sh

# Build with variant model, tag as latest, and push
MODEL_NAME=omniASR_LLM_1B_v2 LATEST_TAG=true PUSH=true bash build.sh

# Build with namespace
NAMESPACE=abc bash build.sh

# Build with namespace and push
NAMESPACE=abc PUSH=true bash build.sh
```

`build.sh` also reads values from `.env` when they are not already set in the shell.

**Build script options:**

- `MODEL_NAME` - Name of the standard model to build (default: value from `.env`, otherwise `omniASR_LLM_300M_v2`)
- `ZERO_SHOT_MODEL_NAME` - Optional dedicated zero-shot model to preload alongside the standard model
- `NAMESPACE` - Namespace/registry prefix for the image name (optional). If provided, images will be tagged as `NAMESPACE/omniasr-server`. If not provided, defaults to `omniasr-server`
- `LATEST_TAG` - Set to `"true"` to also tag the image as `latest` (default: `false`)
- `PUSH` - Set to `"true"` to push the image to the registry after building (default: `false`)

The image will be tagged as `<namespace>/omniasr-server:cu128-pt280-<model-suffix>`. When `ZERO_SHOT_MODEL_NAME` is set, the tag gets a `-with-zs` suffix.

### Manual build

You can also build manually using Docker:

```bash
docker build \
  --build-arg MODEL_NAME=omniASR_CTC_300M_v2 \
  --build-arg ZERO_SHOT_MODEL_NAME=omniASR_LLM_7B_ZS \
  -t omniasr-server .
```

Then, run with GPU support:

```bash
docker run --gpus all -p 8080:8080 omniasr-server
```

## API Usage

The API is (somewhat) compatible with OpenAI's Whisper transcription endpoint. Some parameters such as `model` are informational only; the backend routes requests to the configured standard or zero-shot pipeline based on the endpoint you call.

Long audio is supported by decoding the upload once, splitting it with Silero VAD into speech-aware chunks, and sending those chunks through Omnilingual-ASR in batches. This keeps each chunk below the upstream 40-second model limit without imposing a hard request-length cap at the API layer.

Uploads are decoded with PyAV, similar to `speaches`/`faster-whisper`, so the server accepts a much wider range of container and codec combinations than a libsndfile-only decode path. Raw `audio/pcm` and `audio/raw` uploads are also accepted as signed 16-bit little-endian mono PCM.

### Transcribe Audio

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "model=omniASR_CTC_300M_v2"
```

### With Language Hint

This works for the LLM variants only. For CTC and W2V, the `language` parameter is ignored.

**ISO 639-1 (OpenAI API native)**

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=omniASR_LLM_1B_v2" \
  -F "language=en"
```

**ISO 639-3 / Script (Omnilingual-ASR native)**

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=omniASR_LLM_1B_v2" \
  -F "language=eng_Latn"
```

Languages are mapped heuristically from ISO 639-1 (Whisper's API) to Omnilingual-ASR's format. See how it's mapped in [`app/languages.py`](app/languages.py). For the best results, use Omnilingual-ASR's language codes.

### Zero-Shot Transcription

Zero-shot inference uses a dedicated endpoint and a dedicated `*_ZS` model configured via `ZERO_SHOT_MODEL_NAME`. Each request must include one or more aligned context audio/text pairs. Send them as repeated multipart fields named `context_files` and `context_texts` in the same order:

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions/zero-shot \
  -F "file=@target.wav" \
  -F "context_files=@example1.wav" \
  -F "context_texts=example one transcript" \
  -F "context_files=@example2.wav" \
  -F "context_texts=example two transcript"
```

The server reuses those context examples for each VAD chunk, calls `transcribe_with_context()` under the hood, and forces `batch_size=1` for zero-shot requests to match the upstream model guidance.

If a single `context_file` is longer than 40 seconds, the server now splits it into fixed 40-second windows before zero-shot inference. Because the incoming request contains only one transcript per file, transcript alignment across those windows is heuristic; shorter, already-aligned context samples still produce better quality.

### Response Formats

**JSON (default)**
```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions -F "file=@audio.wav"
```
```json
{"text": "Hello, world!"}
```

**Plain Text**
```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=text"
```

**Verbose JSON With Timestamps**
```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities=segment" \
  -F "timestamp_granularities=word"
```

`timestamp_granularities` is only returned with `response_format=verbose_json`. In this server, segment timestamps are derived from Silero VAD chunks and word timestamps are approximate timings distributed across the words inside each chunk rather than native model alignments.

### Python Client

```bash
uv run scripts/openai_client.py
```

See the [openai_client.py](scripts/openai_client.py) code. It's pretty straightforward.


## Configuration


### Environment Variables

The server automatically loads `.env` from the repository root.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `omniASR_CTC_300M_v2` | Standard model used by `/v1/audio/transcriptions` |
| `ZERO_SHOT_MODEL_NAME` | unset | Optional dedicated `*_ZS` model used by `/v1/audio/transcriptions/zero-shot` |
| `OMNILINGUAL_PORT` | `8080` | Server port |
| `OMNILINGUAL_HOST` | `0.0.0.0` | Server host |
| `OMNILINGUAL_ROOT_PATH` | `/omni-asr` | ASGI `root_path` used when the API is published behind a URL prefix |
| `OMNILINGUAL_DEVICE` | `auto` | Device selection: `auto`, `cpu`, `cuda`, or `mps` |
| `OMNILINGUAL_BATCH_SIZE` | `4` | Maximum number of decoded chunks sent to the model in one inference batch |
| `OMNILINGUAL_PRELOAD_ZERO_SHOT` | `false` | When `true`, eagerly loads the zero-shot pipeline during startup instead of on first zero-shot request |
| `OMNILINGUAL_CHUNK_MAX_SECONDS` | `30` | Maximum chunk length after Silero VAD splitting. Must stay below the model's 40 second limit |
| `OMNILINGUAL_VAD_THRESHOLD` | `0.5` | Silero speech threshold. Higher values make speech detection stricter |
| `OMNILINGUAL_VAD_NEG_THRESHOLD` | unset | Optional Silero silence threshold override. Blank uses Silero's auto fallback |
| `OMNILINGUAL_VAD_MIN_SPEECH_MS` | `250` | Minimum speech duration kept as a speech segment |
| `OMNILINGUAL_VAD_MIN_SILENCE_MS` | `300` | Silence duration required to close a speech segment |
| `OMNILINGUAL_VAD_SPEECH_PAD_MS` | `200` | Padding added around detected speech segments before chunk merge/split |

When `OMNILINGUAL_DEVICE=cuda`, Silero VAD also runs on CUDA so both chunk planning and ASR stay on the GPU path.

Additional variables relevant to Docker/build workflows:

| Variable | Default | Description |
|----------|---------|-------------|
| `NAMESPACE` | unset | Optional registry/namespace prefix |
| `LATEST_TAG` | `false` | Also tag the image as `latest` |
| `PUSH` | `false` | Push the built image after build |
| `FAIRSEQ2_CACHE_DIR` | `/models/fairseq2/assets` | Optional fairseq cache location inside the container |

### Changing the Model

See [Omnilingual-ASR's GitHub page](https://github.com/facebookresearch/omnilingual-asr/tree/main?tab=readme-ov-file#model-architectures) for a list of available models.

The current API surface supports a dual-model setup:

- `MODEL_NAME` for standard CTC, LLM, or LLM Unlimited inference through `/v1/audio/transcriptions`
- `ZERO_SHOT_MODEL_NAME` for zero-shot `*_ZS` inference through `/v1/audio/transcriptions/zero-shot`

Set `ZERO_SHOT_MODEL_NAME` only to a `*_ZS` card such as `omniASR_LLM_7B_ZS`. The standard `MODEL_NAME` must stay on a non-zero-shot model.

You can specify the model either at build time or at runtime:

**At build time (recommended):**

```bash
# Build with a specific standard model
MODEL_NAME=omniASR_LLM_1B_v2 bash build.sh

# Build with both standard and zero-shot models preloaded
MODEL_NAME=omniASR_CTC_300M_v2 ZERO_SHOT_MODEL_NAME=omniASR_LLM_7B_ZS bash build.sh

# Then run the container
docker run --gpus all -p 8080:8080 omniasr-server:cu128-pt280-llm-1b-v2
```

**At runtime:**

```bash
# Run with a different standard model and enable zero-shot too
docker run --gpus all -p 8080:8080 \
  -e MODEL_NAME=omniASR_CTC_1B_v2 \
  -e ZERO_SHOT_MODEL_NAME=omniASR_LLM_7B_ZS \
  omniasr-server
```

**When running locally:**

```bash
MODEL_NAME=omniASR_CTC_1B_v2 \
ZERO_SHOT_MODEL_NAME=omniASR_LLM_7B_ZS \
uv run python main.py
```

**NOTE:** When running locally, on the first run, `fairseq` will download the weights and cache it to your device. Subsequent runs only loads the cached weights.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/transcriptions` | POST | Transcribe audio with the standard model |
| `/v1/audio/transcriptions/zero-shot` | POST | Transcribe audio with the dedicated zero-shot model and context examples |
| `/v1/models` | GET | List the configured model cards |
| `/health-check` | GET | Health check |

## License

This server code is MIT licensed. The Omnilingual ASR models are released under Apache 2.0 by Meta.
