# Builder stage - has build tools
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS builder

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
COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev --frozen

# Final stage - runtime only
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/pyproject.toml /app/
COPY --from=builder /app/uv.lock /app/

ENV FAIRSEQ2_CACHE_DIR=/models/fairseq2/assets

COPY app/ app/
COPY main.py main.py
COPY scripts/ scripts/

EXPOSE 8080

CMD ["uv", "run", "--no-dev", "--frozen", "main.py"]
