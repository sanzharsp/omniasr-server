"""Pre-download configured model assets into the fairseq cache."""

from fairseq2.assets import AssetDownloadManager, AssetStore
from fairseq2.data.tokenizers.ref import resolve_tokenizer_reference
from fairseq2.runtime.dependency import get_dependency_resolver

from app.config import ALIGNMENT_MODEL_NAME, MODEL_NAME, ZERO_SHOT_MODEL_NAME


def _configured_model_names() -> list[str]:
    """Return configured model names without duplicates."""

    model_names = [MODEL_NAME]
    if ZERO_SHOT_MODEL_NAME and ZERO_SHOT_MODEL_NAME not in model_names:
        model_names.append(ZERO_SHOT_MODEL_NAME)
    if ALIGNMENT_MODEL_NAME and ALIGNMENT_MODEL_NAME not in model_names:
        model_names.append(ALIGNMENT_MODEL_NAME)

    return model_names


def _preload_one_model(
    *,
    model_name: str,
    asset_store: AssetStore,
    download_manager: AssetDownloadManager,
) -> None:
    """Download a model checkpoint and its tokenizer."""

    print(f"Pre-downloading model: {model_name}")

    card = asset_store.retrieve_card(model_name)

    checkpoint_uri = card.field("checkpoint").as_uri()
    print(f"Downloading model checkpoint from: {checkpoint_uri}")
    checkpoint_path = download_manager.download_model(
        checkpoint_uri,
        model_name,
        progress=True,
    )
    print(f"Model checkpoint downloaded to: {checkpoint_path}")

    tokenizer_card = resolve_tokenizer_reference(asset_store, card)
    tokenizer_uri = tokenizer_card.field("tokenizer").as_uri()
    print(f"Downloading tokenizer from: {tokenizer_uri}")
    tokenizer_path = download_manager.download_tokenizer(
        tokenizer_uri,
        tokenizer_card.name,
        progress=True,
    )
    print(f"Tokenizer downloaded to: {tokenizer_path}")

    print(f"All assets for {model_name} downloaded successfully!")


def preload_model():
    """Download all configured model assets into cache."""

    resolver = get_dependency_resolver()
    asset_store = resolver.resolve(AssetStore)
    download_manager = resolver.resolve(AssetDownloadManager)

    for model_name in _configured_model_names():
        _preload_one_model(
            model_name=model_name,
            asset_store=asset_store,
            download_manager=download_manager,
        )


if __name__ == "__main__":
    preload_model()
