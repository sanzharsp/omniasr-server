# Omnilingual-ASR Model Server

A FastAPI-based ASR model server for [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) with an OpenAI Whisper-compatible API.

## Quick Start

### Local Development

```bash
# Install dependencies with uv
uv sync

# Run the server
uv run python main.py
```

## Building a Docker Image

This repo builds the image with CUDA 12.6 and PyTorch 2.8.0. To build against a different CUDA version, you need to update the sources and indices in [pyproject.toml](pyproject.toml).

### Helpful resources

- Supported combinations of CUDA, PyTorch, and Python for `fairseq2`: https://github.com/facebookresearch/fairseq2?tab=readme-ov-file#variants
- Organizing sources and indices: https://docs.astral.sh/uv/concepts/indexes/

### Using the build script

The [`build.sh`](build.sh) script is a good place to start your own builds:

```bash
# Build
bash build.sh

# Build, tag as latest, and push
LATEST_TAG=true PUSH=true bash build.sh

# Build with namespace
NAMESPACE=abc bash build.sh

# Build with namespace and push
NAMESPACE=abc PUSH=true bash build.sh
```

**Build script options:**

- `NAMESPACE` - Namespace/registry prefix for the image name (optional). If provided, images will be tagged as `NAMESPACE/omniasr-server`. If not provided, defaults to `omniasr-server`
- `LATEST_TAG` - Set to `"true"` to also tag the image as `latest` (default: `false`)
- `PUSH` - Set to `"true"` to push the image to the registry after building (default: `false`)

The image will be tagged as `<namespace>/omniasr-server:cu126-pt280`. Because the image ships without weights, a single image works for all model variants.

### Manual build

You can also build manually using Docker:

```bash
docker build -t omniasr-server .
```

Then, run with GPU support:

```bash
docker run --gpus all -p 8080:8080 -e MODEL_NAME=omniASR_CTC_300M_v2 omniasr-server
```

I'm open to 💡 on how to streamline the build process so I can build for multiple CUDA and PyTorch versions.

## API Usage

The API is (somewhat) compatible with OpenAI's Whisper transcription endpoint. Some parameters (like `model`) are ignored (for now!) since the server only hosts one model and Omnilingual-ASR doesn't have all the features of Whisper.

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

### Python Client

```bash
uv run scripts/openai_client.py
```

See the [openai_client.py](scripts/openai_client.py) code. It's pretty straightforward.


## Configuration


### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `omniASR_CTC_300M_v2` | Model to use for transcription |
| `MODEL_CHECKPOINT_URL` | _(fairseq2 registry)_ | Custom URL for the model checkpoint (see [Airgap / self-hosted S3](#airgap--self-hosted-s3)) |
| `MODEL_TOKENIZER_URL` | _(fairseq2 registry)_ | Custom URL for the model tokenizer (see [Airgap / self-hosted S3](#airgap--self-hosted-s3)) |
| `OMNILINGUAL_PORT` | `8080` | Server port |
| `OMNILINGUAL_HOST` | `0.0.0.0` | Server host |

### Changing the Model

See [Omnilingual-ASR's GitHub page](https://github.com/facebookresearch/omnilingual-asr/tree/main?tab=readme-ov-file#model-architectures) for a list of available models.

The Docker image ships **without model weights**. Weights are downloaded on the first startup and cached in `FAIRSEQ2_CACHE_DIR` (`/models/fairseq2/assets` inside the container). Subsequent starts load from that cache.

```bash
# Run — model is downloaded on first start
docker run --gpus all -p 8080:8080 \
  -e MODEL_NAME=omniASR_CTC_1B_v2 \
  omniasr-server:cu126-pt280
```

Persist the cache across restarts with a volume mount:

```bash
docker run --gpus all -p 8080:8080 \
  -e MODEL_NAME=omniASR_CTC_1B_v2 \
  -v /path/to/model-cache:/models/fairseq2/assets \
  omniasr-server:cu126-pt280
```

**When running locally:**

```bash
MODEL_NAME=omniASR_CTC_1B_v2 uv run python main.py
```

On the first run fairseq2 downloads the weights and caches them. Subsequent runs load from the cache.

### Airgap / self-hosted S3

In environments where the default fairseq2 CDN (`dl.fbaipublicfiles.com`) is unreachable, provide direct download URLs via environment variables:

```bash
docker run --gpus all -p 8080:8080 \
  -e MODEL_NAME=omniASR_CTC_300M_v2 \
  -e MODEL_CHECKPOINT_URL=https://s3.internal/models/omniASR_CTC_300M_v2/checkpoint.pt \
  -e MODEL_TOKENIZER_URL=https://s3.internal/models/tokenizer.model \
  omniasr-server:cu126-pt280
```

- `MODEL_CHECKPOINT_URL` — direct HTTPS (or S3) URL for the model checkpoint.
- `MODEL_TOKENIZER_URL` — direct HTTPS (or S3) URL for the SentencePiece tokenizer. Required only when the tokenizer CDN URL is also unreachable.

Both variables are optional. If omitted, fairseq2 falls back to its default registry URLs.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/transcriptions` | POST | Transcribe audio file |
| `/v1/models` | GET | List the deployed model |
| `/health-check` | GET | Health check |

## License

This server code is MIT licensed. The Omnilingual ASR models are released under Apache 2.0 by Meta.

