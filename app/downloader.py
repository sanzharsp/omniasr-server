"""Model download utilities for Omnilingual-ASR server.

Downloads model weights at server startup. Supports custom checkpoint and
tokenizer URLs via fairseq2 user-asset YAML overrides — intended for airgap
or self-hosted S3 environments.
"""

import logging
from pathlib import Path

from fairseq2.assets import AssetDownloadManager, AssetStore
from fairseq2.data.tokenizers.ref import resolve_tokenizer_reference
from fairseq2.runtime.dependency import get_dependency_resolver

from app.config import MODEL_CHECKPOINT_URL, MODEL_NAME, MODEL_TOKENIZER_URL

logger = logging.getLogger(__name__)

# fairseq2 reads user-defined asset cards from this directory on startup.
_USER_ASSET_DIR = Path.home() / ".config" / "fairseq2" / "assets"
_OVERRIDE_FILE = _USER_ASSET_DIR / "omniasr_server_override.yaml"


def _write_asset_overrides(
    checkpoint_url: str | None,
    tokenizer_name: str | None,
    tokenizer_url: str | None,
) -> None:
    """Write (or refresh) the fairseq2 user-asset override YAML."""
    _USER_ASSET_DIR.mkdir(parents=True, exist_ok=True)

    entries: list[str] = []

    if checkpoint_url:
        entries.append(f"- name: {MODEL_NAME}\n  checkpoint: '{checkpoint_url}'")
        logger.info("Overriding %s checkpoint URL → %s", MODEL_NAME, checkpoint_url)

    if tokenizer_name and tokenizer_url:
        entries.append(f"- name: {tokenizer_name}\n  tokenizer: '{tokenizer_url}'")
        logger.info("Overriding %s tokenizer URL → %s", tokenizer_name, tokenizer_url)

    if entries:
        _OVERRIDE_FILE.write_text("\n".join(entries) + "\n")
        logger.debug("Wrote asset overrides to %s", _OVERRIDE_FILE)


def _make_resolver():
    return get_dependency_resolver()


def ensure_model_available() -> None:
    """Download model weights at startup if not already cached.

    This is idempotent: fairseq2's download manager skips files that are
    already present in FAIRSEQ2_CACHE_DIR (e.g. from a mounted volume).
    """
    logger.info("Ensuring model %s is available...", MODEL_NAME)

    # Write the checkpoint override *before* the first resolver call so that
    # fairseq2 picks it up when it initialises the asset store.
    if MODEL_CHECKPOINT_URL:
        _write_asset_overrides(MODEL_CHECKPOINT_URL, None, None)

    resolver = _make_resolver()
    asset_store = resolver.resolve(AssetStore)
    download_manager = resolver.resolve(AssetDownloadManager)

    card = asset_store.retrieve_card(MODEL_NAME)

    # Discover the tokenizer card name (metadata-only — no download yet).
    tokenizer_card = resolve_tokenizer_reference(asset_store, card)
    tokenizer_name = tokenizer_card.name

    # Now we know the tokenizer card name; write the tokenizer override and
    # re-initialise the resolver so the new YAML is honoured.
    if MODEL_TOKENIZER_URL:
        _write_asset_overrides(MODEL_CHECKPOINT_URL, tokenizer_name, MODEL_TOKENIZER_URL)
        resolver = _make_resolver()
        asset_store = resolver.resolve(AssetStore)
        download_manager = resolver.resolve(AssetDownloadManager)
        card = asset_store.retrieve_card(MODEL_NAME)
        tokenizer_card = resolve_tokenizer_reference(asset_store, card)

    checkpoint_uri = card.field("checkpoint").as_uri()
    logger.info("Downloading checkpoint: %s", checkpoint_uri)
    download_manager.download_model(checkpoint_uri, MODEL_NAME, progress=True)

    tokenizer_uri = tokenizer_card.field("tokenizer").as_uri()
    logger.info("Downloading tokenizer: %s", tokenizer_uri)
    download_manager.download_tokenizer(tokenizer_uri, tokenizer_card.name, progress=True)

    logger.info("Model %s is ready.", MODEL_NAME)
