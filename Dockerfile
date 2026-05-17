ARG MODEL_NAME
ARG ZERO_SHOT_MODEL_NAME
ARG ALIGNMENT_MODEL_NAME

# Builder stage - has build tools
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    libsndfile1 \
    git \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml .

RUN uv sync --no-dev

# Final stage - runtime only
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libsndfile1 \
    curl \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && groupadd --system --gid 1000 appuser \
    && useradd --system --uid 1000 --gid appuser --home /home/appuser --create-home appuser

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/pyproject.toml /app/

ARG MODEL_NAME
ARG ZERO_SHOT_MODEL_NAME
ARG ALIGNMENT_MODEL_NAME
ENV FAIRSEQ2_CACHE_DIR=/models/fairseq2/assets
ENV MODEL_NAME=${MODEL_NAME}
ENV ZERO_SHOT_MODEL_NAME=${ZERO_SHOT_MODEL_NAME}
ENV ALIGNMENT_MODEL_NAME=${ALIGNMENT_MODEL_NAME}

COPY app/ app/
COPY main.py main.py
COPY scripts/ scripts/

# Pre-download model to cache during build
RUN PYTHONPATH=/app uv run --no-dev scripts/preload.py \
    && mkdir -p /models/lid \
    && curl -fsSL -o /models/lid/lid.176.bin \
        https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
    && chmod -R a+rX /models \
    && chown -R appuser:appuser /app /models

USER appuser

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${OMNILINGUAL_PORT:-8081}${OMNILINGUAL_ROOT_PATH:-}/healthz || exit 1

# OCI image labels — surfaced by `docker inspect` and GitHub package UI.
LABEL org.opencontainers.image.title="omniasr-server" \
      org.opencontainers.image.description="Omnilingual-ASR server with VAD, CTC forced alignment, and per-chunk LID." \
      org.opencontainers.image.licenses="MIT"

# tini handles PID 1 duties (signal forwarding, zombie reaping) so SIGTERM
# from `docker stop` cleanly reaches uvicorn instead of being swallowed.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uv", "run", "--no-dev", "main.py"]
