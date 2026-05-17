"""Word-level alignment strategies for transcribed speech chunks."""

from __future__ import annotations

import logging
from threading import Lock
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch
import torchaudio.functional as taf

from app.audio import WordTimestamp

if TYPE_CHECKING:
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

logger = logging.getLogger(__name__)


class WordAligner(Protocol):
    """Strategy that maps a chunk transcript to word-level timestamps."""

    def align(
        self,
        *,
        text: str,
        waveform: np.ndarray,
        sample_rate: int,
        chunk_start_seconds: float,
        chunk_end_seconds: float,
    ) -> list[WordTimestamp]:
        """Return word timestamps in absolute seconds for the chunk's transcript."""


class HeuristicWordAligner:
    """Distribute chunk duration across words proportionally to their letter weight.

    Fast, model-free fallback. Accuracy degrades when a chunk contains internal
    pauses or words with very different articulation speed.
    """

    def align(
        self,
        *,
        text: str,
        waveform: np.ndarray,
        sample_rate: int,
        chunk_start_seconds: float,
        chunk_end_seconds: float,
    ) -> list[WordTimestamp]:
        del waveform, sample_rate

        words = text.split()
        if not words:
            return []

        total_duration = max(0.0, chunk_end_seconds - chunk_start_seconds)
        if total_duration == 0.0:
            return [
                WordTimestamp(
                    word=word,
                    start_seconds=chunk_start_seconds,
                    end_seconds=chunk_end_seconds,
                )
                for word in words
            ]

        weights = [max(sum(char.isalnum() for char in word), 1) for word in words]
        total_weight = sum(weights)

        current_start = chunk_start_seconds
        timed_words: list[WordTimestamp] = []

        for index, (word, weight) in enumerate(zip(words, weights, strict=True)):
            if index == len(words) - 1:
                current_end = chunk_end_seconds
            else:
                current_end = current_start + (total_duration * (weight / total_weight))

            timed_words.append(
                WordTimestamp(
                    word=word,
                    start_seconds=current_start,
                    end_seconds=current_end,
                )
            )
            current_start = current_end

        return timed_words


class CTCForcedAligner:
    """Per-frame CTC forced alignment using a separate Wav2Vec2 CTC model.

    The transcription comes from another pipeline (e.g. an LLM-conditioned
    model); this aligner only computes word timestamps. The CTC model is
    used purely as an alignment grid.

    Frame stride is recovered empirically per chunk (samples_in / frames_out),
    so the aligner works without hard-coded model internals.
    """

    def __init__(
        self,
        *,
        pipeline: ASRInferencePipeline,
        fallback: WordAligner | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.model = pipeline.model
        self.tokenizer = pipeline.tokenizer
        self.device = pipeline.device
        self.dtype = pipeline.dtype
        self.fallback: WordAligner = fallback or HeuristicWordAligner()

        blank_idx = getattr(self.tokenizer.vocab_info, "pad_idx", None)
        self.blank_idx = int(blank_idx) if blank_idx is not None else 0

        self._token_decoder = self.tokenizer.create_decoder(skip_special_tokens=False)
        self._token_encoder = self.tokenizer.create_encoder()
        self._lock = Lock()

    def align(
        self,
        *,
        text: str,
        waveform: np.ndarray,
        sample_rate: int,
        chunk_start_seconds: float,
        chunk_end_seconds: float,
    ) -> list[WordTimestamp]:
        if not text.strip() or waveform.size == 0:
            return self.fallback.align(
                text=text,
                waveform=waveform,
                sample_rate=sample_rate,
                chunk_start_seconds=chunk_start_seconds,
                chunk_end_seconds=chunk_end_seconds,
            )

        try:
            return self._align_unsafe(
                text=text,
                waveform=waveform,
                sample_rate=sample_rate,
                chunk_start_seconds=chunk_start_seconds,
                chunk_end_seconds=chunk_end_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CTC forced alignment failed (%s); falling back to heuristic",
                exc,
            )
            return self.fallback.align(
                text=text,
                waveform=waveform,
                sample_rate=sample_rate,
                chunk_start_seconds=chunk_start_seconds,
                chunk_end_seconds=chunk_end_seconds,
            )

    def _align_unsafe(
        self,
        *,
        text: str,
        waveform: np.ndarray,
        sample_rate: int,
        chunk_start_seconds: float,
        chunk_end_seconds: float,
    ) -> list[WordTimestamp]:
        token_ids = self._encode_text(text)
        if not token_ids:
            return []

        word_boundaries = self._word_boundaries(token_ids)
        if not word_boundaries:
            return []

        log_probs, num_frames, num_samples = self._encode_waveform(waveform, sample_rate)
        if num_frames == 0:
            raise RuntimeError("CTC encoder returned zero frames")

        targets = torch.tensor([token_ids], dtype=torch.long, device=log_probs.device)
        input_lengths = torch.tensor([num_frames], dtype=torch.long, device=log_probs.device)
        target_lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=log_probs.device)

        alignment, _ = taf.forced_align(
            log_probs,
            targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=self.blank_idx,
        )
        path = alignment[0].tolist()
        spans = self._collapse_path_to_token_spans(path, expected=len(token_ids))
        if len(spans) != len(token_ids):
            raise RuntimeError(
                f"forced_align produced {len(spans)} spans for {len(token_ids)} targets"
            )

        stride_seconds = (num_samples / num_frames) / sample_rate
        timed_words: list[WordTimestamp] = []
        for word, start_tok, end_tok in word_boundaries:
            start_frame = spans[start_tok][0]
            end_frame = spans[end_tok][1]
            raw_start = chunk_start_seconds + start_frame * stride_seconds
            raw_end = chunk_start_seconds + (end_frame + 1) * stride_seconds
            # Clamp BOTH bounds into [chunk_start, chunk_end] before enforcing
            # end >= start — otherwise a frame past the logical end produces
            # end < start (the bug seen in long-tail chunks).
            start_clamped = max(chunk_start_seconds, min(raw_start, chunk_end_seconds))
            end_clamped = max(start_clamped, min(raw_end, chunk_end_seconds))
            timed_words.append(
                WordTimestamp(
                    word=word,
                    start_seconds=start_clamped,
                    end_seconds=end_clamped,
                )
            )
        return timed_words

    def _encode_text(self, text: str) -> list[int]:
        encoded = self._token_encoder(text)
        if hasattr(encoded, "tolist"):
            ids = encoded.tolist()
        else:
            ids = list(encoded)
        # Drop special tokens (BOS/EOS/pad) — they have no acoustic correspondence.
        special = self._special_token_ids()
        return [int(i) for i in ids if int(i) not in special]

    def _special_token_ids(self) -> set[int]:
        info = self.tokenizer.vocab_info
        specials: set[int] = set()
        for attr in ("pad_idx", "bos_idx", "eos_idx", "unk_idx", "boa_idx", "eoa_idx"):
            value = getattr(info, attr, None)
            if value is not None:
                specials.add(int(value))
        return specials

    def _word_boundaries(self, token_ids: list[int]) -> list[tuple[str, int, int]]:
        """Group token ids into word spans (word_text, first_tok_idx, last_tok_idx).

        SentencePiece convention: a token whose decoded form starts with a
        leading space (or `▁`) marks the beginning of a new word.
        """
        words: list[tuple[str, int, int]] = []
        current_chars: list[str] = []
        current_start = 0
        for index, token_id in enumerate(token_ids):
            piece = self._decode_token_piece(token_id)
            starts_word = piece.startswith(" ") or piece.startswith("▁") or index == 0
            stripped = piece.lstrip(" ▁")
            if starts_word and current_chars:
                words.append(("".join(current_chars).strip(), current_start, index - 1))
                current_chars = []
                current_start = index
            current_chars.append(stripped)
        if current_chars:
            words.append(("".join(current_chars).strip(), current_start, len(token_ids) - 1))
        return [w for w in words if w[0]]

    def _decode_token_piece(self, token_id: int) -> str:
        tensor = torch.tensor([token_id], dtype=torch.long)
        decoded = self._token_decoder(tensor)
        if isinstance(decoded, list):
            decoded = decoded[0] if decoded else ""
        return str(decoded)

    def _collapse_path_to_token_spans(
        self,
        path: list[int],
        *,
        expected: int,
    ) -> list[tuple[int, int]]:
        """Walk the forced-align path, returning (start, end) frame indices for
        each non-blank run. Each run corresponds to the next target token."""

        spans: list[tuple[int, int]] = []
        index = 0
        while index < len(path) and len(spans) < expected:
            token = path[index]
            if token == self.blank_idx:
                index += 1
                continue
            start = index
            while index < len(path) and path[index] == token:
                index += 1
            spans.append((start, index - 1))
        return spans

    def _encode_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> tuple[torch.Tensor, int, int]:
        """Run the CTC encoder on a normalized waveform; return (log_probs, T, S)."""

        from fairseq2.nn.batch_layout import BatchLayout

        builder = self.pipeline._build_audio_wavform_pipeline(
            [{"waveform": waveform.astype(np.float32, copy=False), "sample_rate": sample_rate}]
        )
        prepared = list(builder.and_return())
        if not prepared:
            raise RuntimeError("Audio pipeline produced no output")
        audio_tensor = prepared[0]
        if audio_tensor.dim() == 1:
            source_seqs = audio_tensor.unsqueeze(0)
        else:
            source_seqs = audio_tensor
        source_seqs = source_seqs.to(self.device, self.dtype)

        batch_layout = BatchLayout(
            source_seqs.shape,
            seq_lens=[source_seqs.shape[1]],
            device=source_seqs.device,
        )

        with self._lock, torch.inference_mode():
            logits, bl_out = self.model(source_seqs, batch_layout)

        seq_len = int(bl_out.seq_lens[0]) if bl_out.seq_lens is not None else logits.shape[1]
        log_probs = torch.log_softmax(logits[:, :seq_len, :].float(), dim=-1)
        return log_probs, seq_len, int(source_seqs.shape[1])
