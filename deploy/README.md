# Production deployment

Single-VM with GPU. The `Docker CI/CD` workflow
(`.github/workflows/cd.yml`) runs on the self-hosted runner: push to `dev`
→ **build** job runs `docker compose build` locally (no registry) → **deploy**
job recreates the `omniasr` container and waits for `/readyz` before reporting
success. Telegram is notified at build start / build failure / deploy start /
success / failure.

## One-time host setup

```bash
# 1. Docker + NVIDIA toolkit (Ubuntu 24.04 LTS)
curl -fsSL https://get.docker.com | sh
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 2. Caddy for TLS
sudo apt-get install -y caddy

# 3. Register this host as a GitHub Actions self-hosted runner
#    (Settings → Actions → Runners). Its label must match `runs-on` in
#    .github/workflows/cd.yml. The image is built here, so no registry login.

# 4. Clone the repo (the runner needs the full repo to build the image)
git clone https://github.com/sanzharsp/omniasr-server.git
cd omniasr-server
cp .env.example .env  # then edit secrets in .env
```

## `.env` checklist for production

| Var | Required | Example |
|---|---|---|
| `OMNILINGUAL_API_KEY` | **yes** | random 32+ char string |
| `OMNILINGUAL_LID_TOKEN` | yes | bearer for the LID service |
| `OMNILINGUAL_LID_URL` | yes | `https://mangisoz.nu.edu.kz/...` |
| `HUGGINGFACE_TOKEN` | yes (build only) | `hf_...` |
| `IMAGE_TAG` | optional | defaults to `latest` |

In production the `.env` is rendered by the `CD` workflow from `.env.example`
plus GitHub secrets/vars — that workflow is the single place to add a prod
secret. The table above is only for manual / first-run deploys on the host.

## Deploy a new version

Normally automatic: push to `dev` → the `Docker CI/CD` workflow builds and
rolls out. To deploy/roll back manually on the host (images are local, so
`SKIP_PULL=1`):

```bash
# Redeploy whatever `latest` points to
SKIP_PULL=1 ./deploy/deploy.sh

# Pin to a specific build
SKIP_PULL=1 IMAGE_TAG=sha-abc123def ./deploy/deploy.sh
```

The script recreates the container and waits for `/readyz` to return 200
before exiting. If readiness never arrives it dumps logs and exits non-zero,
and the deploy job's `Wait for /readyz` step does the same — a failed rollout
shows red in Actions + a Telegram failure alert (no automatic rollback; roll
back by hand below).

## Rollback

The workflow builds `omniasr-server:latest` each run, so to roll back you
re-run the workflow on a known-good commit, or on the host:

```bash
# Inspect what's on the host
docker image ls omniasr-server

# Re-deploy a previous build that still exists locally
SKIP_PULL=1 IMAGE_TAG=<tag> ./deploy/deploy.sh
```

> The workflow prunes images older than ~1 week
> (`docker image prune --filter until=168h`). Bump that window in
> `.github/workflows/cd.yml` if you need a longer rollback horizon.

## Health endpoints (no auth)

- `GET /healthz` — process is alive.
- `GET /readyz` — models are loaded, traffic can flow.

## Auth

Bearer token from `OMNILINGUAL_API_KEY` is required for `POST /v1/*` paths.
Probes and `/docs` stay open. If `OMNILINGUAL_API_KEY` is empty the auth
middleware is a no-op (dev mode) — never deploy that to prod.

```bash
curl -X POST https://asr.example.com/omni-asr/v1/audio/transcriptions \
  -H "Authorization: Bearer $OMNILINGUAL_API_KEY" \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities=word"
```

## Logs

Structured JSON on stdout (`docker compose logs -f omniasr`). Each request
carries a request id propagated as `X-Request-ID` header — include that id
in any bug report.

Rotation is set up via Docker's `json-file` driver: 50MB per file, 5 files
kept, so the daemon caps log usage to ~250MB per container.

## Caddy

```bash
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

That's it — Caddy obtains and renews the certificate automatically.
