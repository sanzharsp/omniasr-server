"""Tests for nearest-anchor LID smoothing.

A chunk is an "anchor" when it is long enough AND its LID confidence is
high enough. Every non-anchor chunk inherits the language of the nearest
anchor on the timeline. On a tie, the earlier anchor wins.
"""

from app.audio import AudioChunk
from app.lid import SmoothingConfig, smooth_chunk_languages
from app.segments import ChunkAnnotation


SR = 16000


def _ann(
    *,
    start_s: float,
    end_s: float,
    text: str,
    language: str | None,
    confidence: float | None,
) -> ChunkAnnotation:
    start = int(start_s * SR)
    end = int(end_s * SR)
    chunk = AudioChunk(
        start_sample=start,
        end_sample=end,
        logical_start_sample=start,
        logical_end_sample=end,
    )
    return ChunkAnnotation(
        chunk=chunk,
        text=text,
        language=language,
        language_confidence=confidence,
    )


def _default_config(**overrides) -> SmoothingConfig:
    return SmoothingConfig(
        short_max_seconds=2.0,
        short_max_words=3,
        anchor_min_confidence=0.7,
        **overrides,
    )


def test_short_chunk_inherits_language_from_nearest_anchor():
    """A 1-word chunk between two anchors of the same language must inherit it."""

    annotations = [
        _ann(start_s=0, end_s=4, text="hello there world four", language="en", confidence=0.9),
        _ann(start_s=4, end_s=4.4, text="ok", language="fr", confidence=0.55),
        _ann(start_s=4.4, end_s=9, text="another full sentence here please", language="en", confidence=0.9),
    ]

    smooth_chunk_languages(annotations, config=_default_config(), sample_rate=SR)

    assert annotations[1].language == "en"
    assert annotations[1].language_inherited is True
    assert annotations[0].language_inherited is False
    assert annotations[2].language_inherited is False


def test_long_chunk_with_low_confidence_is_not_anchor_and_gets_overridden():
    """A long chunk with low LID confidence is unreliable — it must also
    inherit from a nearby anchor instead of being trusted blindly."""

    annotations = [
        # anchor: long enough AND confident
        _ann(start_s=0, end_s=6, text="big english chunk one here", language="en", confidence=0.9),
        # long (5s, 5 words) but low confidence (0.5 < 0.7) — NOT an anchor
        _ann(start_s=6, end_s=11, text="davay capsindik qay vakta zimandayın", language="uz", confidence=0.5),
        # anchor: long enough AND confident
        _ann(start_s=11, end_s=17, text="big english chunk two more here", language="en", confidence=0.9),
    ]

    smooth_chunk_languages(annotations, config=_default_config(), sample_rate=SR)

    assert annotations[1].language == "en"
    assert annotations[1].language_inherited is True


def test_tie_prefers_earlier_anchor():
    """When the short chunk is equidistant from two anchors, the earlier
    anchor wins (speech tends to carry the previous language)."""

    annotations = [
        _ann(start_s=0, end_s=4, text="english one two three four", language="en", confidence=0.9),
        # midpoint = 5.5 — equidistant from anchor 0 (midpoint 2) and 2 (midpoint 9)? no
        # Use perfect symmetry:
        # anchor 0: [0, 4] mid=2.0
        # short:    [4, 5] mid=4.5
        # anchor 2: [5, 9] mid=7.0
        # distances: |4.5-2.0|=2.5, |4.5-7.0|=2.5 → tie, earlier wins
        _ann(start_s=4, end_s=5, text="да", language="ru", confidence=0.4),
        _ann(start_s=5, end_s=9, text="russian one two three four", language="ru", confidence=0.9),
    ]

    smooth_chunk_languages(annotations, config=_default_config(), sample_rate=SR)

    # short at index 1 is equidistant to both anchors but the earlier (en) wins.
    assert annotations[1].language == "en"
    assert annotations[1].language_inherited is True


def test_anchor_arbitrarily_far_still_wins_without_window():
    """Anchor at the very end must rescue a short chunk at the start — no
    arbitrary distance cap (the algorithm has no window)."""

    annotations = [
        _ann(start_s=0, end_s=0.5, text="ok", language="fr", confidence=0.3),
        _ann(start_s=0.5, end_s=1.0, text="да", language="ru", confidence=0.3),
        _ann(start_s=1.0, end_s=1.5, text="hm", language="de", confidence=0.3),
        _ann(start_s=1.5, end_s=2.0, text="oh", language="es", confidence=0.3),
        _ann(start_s=2.0, end_s=10.0, text="big english anchor far away here", language="en", confidence=0.95),
    ]

    smooth_chunk_languages(annotations, config=_default_config(), sample_rate=SR)

    for i in range(4):
        assert annotations[i].language == "en", f"chunk {i} should inherit en"
        assert annotations[i].language_inherited is True
    assert annotations[4].language_inherited is False


def test_all_chunks_short_falls_back_to_request_language():
    """If no anchor exists anywhere, chunks with no detected language
    fall back to the request language. Chunks that already have a (noisy)
    language are left alone."""

    annotations = [
        _ann(start_s=0, end_s=0.5, text="ok", language=None, confidence=None),
        _ann(start_s=0.5, end_s=1.0, text="да", language="ru", confidence=0.3),
    ]

    smooth_chunk_languages(
        annotations,
        config=_default_config(fallback_language="kk"),
        sample_rate=SR,
    )

    assert annotations[0].language == "kk"
    assert annotations[0].language_inherited is True
    # noisy detection is kept — we have nothing better to override it with.
    assert annotations[1].language == "ru"
    assert annotations[1].language_inherited is False


def test_anchor_chunks_themselves_are_never_modified():
    """Anchor chunks keep their detected language even if they disagree
    with each other — only short/unconfident chunks get rewritten."""

    annotations = [
        _ann(start_s=0, end_s=5, text="english anchor block one two", language="en", confidence=0.95),
        _ann(start_s=5, end_s=11, text="kazakh anchor block one two", language="kk", confidence=0.9),
    ]

    smooth_chunk_languages(annotations, config=_default_config(), sample_rate=SR)

    assert annotations[0].language == "en"
    assert annotations[0].language_inherited is False
    assert annotations[1].language == "kk"
    assert annotations[1].language_inherited is False
