"""Async ASR service for Omnilingual-ASR models."""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from dataclasses import replace
from threading import Lock

import torch
from fastapi import FastAPI
from omnilingual_asr.models.inference.pipeline import (
    ASRInferencePipeline,
    ContextExample,
)
from silero_vad import load_silero_vad

from app.aligners import CTCForcedAligner, HeuristicWordAligner, WordAligner
from app.audio import (
    AudioChunk,
    MODEL_MAX_AUDIO_SECONDS,
    TimedTranscriptSegment,
    WordTimestamp,
    build_speech_chunks,
    decode_audio_bytes,
    join_transcript_texts,
)
from app.chunking import ChunkingConfig
from app.config import (
    ALIGNMENT_MODEL_NAME,
    MODEL_NAME,
    OMNILINGUAL_BATCH_SIZE,
    OMNILINGUAL_DEVICE,
    OMNILINGUAL_LID_ANCHOR_MIN_CONFIDENCE,
    OMNILINGUAL_LID_BACKEND,
    OMNILINGUAL_LID_MODEL_PATH,
    OMNILINGUAL_LID_SHORT_MAX_SECONDS,
    OMNILINGUAL_LID_SHORT_MAX_WORDS,
    OMNILINGUAL_LID_TIMEOUT_SECONDS,
    OMNILINGUAL_LID_TOKEN,
    OMNILINGUAL_LID_URL,
    OMNILINGUAL_PRELOAD_ZERO_SHOT,
    ZERO_SHOT_MODEL_NAME,
)
from app.lid import (
    LanguageDetector,
    NullLanguageDetector,
    SmoothingConfig,
    load_language_detector,
    smooth_chunk_languages,
)
from app.segments import ChunkAnnotation
from app.languages import map_whisper_to_omnilingual

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UploadedContextExample:
    """A raw request context example before decode."""

    audio_bytes: bytes
    text: str
    content_type: str | None = None
    filename: str | None = None


@dataclass(frozen=True)
class DetailedTranscription:
    """A transcription result enriched with timing metadata."""

    text: str
    duration_seconds: float
    language: str | None
    segments: list[TimedTranscriptSegment]
    words: list[WordTimestamp]


class OmnilingualASRService:
    """Async ASR service wrapping standard and zero-shot Omnilingual-ASR pipelines."""

    def __init__(
        self,
        word_aligner: WordAligner | None = None,
        language_detector: LanguageDetector | None = None,
    ):
        self.standard_model_name = MODEL_NAME
        self.zero_shot_model_name = ZERO_SHOT_MODEL_NAME
        self.alignment_model_name = ALIGNMENT_MODEL_NAME

        self.standard_pipeline: ASRInferencePipeline | None = None
        self.zero_shot_pipeline: ASRInferencePipeline | None = None
        self.alignment_pipeline: ASRInferencePipeline | None = None

        self.device: str | None = None
        self.dtype: torch.dtype | None = None

        self.vad_model: torch.nn.Module | None = None
        self.vad_device = torch.device("cpu")

        self._word_aligner_override = word_aligner
        self.word_aligner: WordAligner = word_aligner or HeuristicWordAligner()

        self._language_detector_override = language_detector
        self.language_detector: LanguageDetector = (
            language_detector or NullLanguageDetector()
        )
        self.smoothing_config = SmoothingConfig(
            short_max_seconds=OMNILINGUAL_LID_SHORT_MAX_SECONDS,
            short_max_words=OMNILINGUAL_LID_SHORT_MAX_WORDS,
            anchor_min_confidence=OMNILINGUAL_LID_ANCHOR_MIN_CONFIDENCE,
        )

        self.runtime_lock = Lock()
        self.pipeline_lock = Lock()
        self.vad_lock = Lock()

    def _cuda_arch(self) -> str | None:
        """Return the current CUDA architecture in torch arch-list format."""

        if not torch.cuda.is_available():
            return None

        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}"

    def _cuda_supported(self) -> bool:
        """Check whether this torch build can execute kernels on the current GPU."""

        current_arch = self._cuda_arch()
        if current_arch is None:
            return False

        return current_arch in torch.cuda.get_arch_list()

    def _select_device(self) -> str:
        """Select the best available device, respecting explicit overrides."""

        if OMNILINGUAL_DEVICE not in {"auto", "cpu", "cuda", "mps"}:
            raise RuntimeError(
                "Invalid OMNILINGUAL_DEVICE value. Use one of: auto, cpu, cuda, mps."
            )

        if OMNILINGUAL_DEVICE == "cpu":
            return "cpu"

        if OMNILINGUAL_DEVICE == "mps":
            if torch.backends.mps.is_available():
                return "mps"

            raise RuntimeError(
                "OMNILINGUAL_DEVICE=mps requested, but MPS is not available."
            )

        if OMNILINGUAL_DEVICE == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "OMNILINGUAL_DEVICE=cuda requested, but CUDA is not available."
                )

            if not self._cuda_supported():
                current_arch = self._cuda_arch()
                supported_arches = " ".join(torch.cuda.get_arch_list())
                device_name = torch.cuda.get_device_name(0)

                raise RuntimeError(
                    "OMNILINGUAL_DEVICE=cuda requested, but the installed PyTorch "
                    f"build cannot run on {device_name} ({current_arch}). Supported "
                    f"architectures: {supported_arches}. Install a newer PyTorch "
                    "build or use OMNILINGUAL_DEVICE=cpu."
                )

            return "cuda"

        if torch.cuda.is_available():
            if self._cuda_supported():
                return "cuda"

            current_arch = self._cuda_arch()
            supported_arches = " ".join(torch.cuda.get_arch_list())
            device_name = torch.cuda.get_device_name(0)

            logger.warning(
                "CUDA device %s (%s) detected, but the installed PyTorch build only "
                "supports %s. Falling back to CPU.",
                device_name,
                current_arch,
                supported_arches,
            )

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def _select_dtype(self, device: str) -> torch.dtype:
        """Choose a dtype suitable for the selected device."""

        if device == "cuda":
            if torch.cuda.get_device_capability() < (8, 0):
                return torch.float16

            return torch.bfloat16

        if device == "mps":
            return torch.float16

        return torch.float32

    def _select_vad_device(self, device: str) -> torch.device:
        """Run Silero VAD on CUDA when available, otherwise keep it on CPU."""

        if device == "cuda":
            return torch.device("cuda")

        return torch.device("cpu")

    @staticmethod
    def _is_llm_model_name(model_name: str) -> bool:
        """Check if a model card belongs to an LLM-based architecture."""

        return "LLM" in model_name

    @staticmethod
    def _requires_context_examples_for_model(model_name: str) -> bool:
        """Check if a model card belongs to the zero-shot family."""

        return model_name.endswith("_ZS")

    @property
    def is_llm_model(self) -> bool:
        """Check if the standard model supports language conditioning."""

        return self._is_llm_model_name(self.standard_model_name)

    @property
    def requires_context_examples(self) -> bool:
        """Return whether the standard model requires in-context examples."""

        return self._requires_context_examples_for_model(self.standard_model_name)

    @property
    def has_zero_shot_model(self) -> bool:
        """Return whether a dedicated zero-shot model is configured."""

        return self.zero_shot_model_name is not None

    @property
    def configured_model_names(self) -> list[str]:
        """Return the configured model cards in API listing order."""

        model_names = [self.standard_model_name]
        if self.zero_shot_model_name and self.zero_shot_model_name not in model_names:
            model_names.append(self.zero_shot_model_name)

        return model_names

    def _validate_model_configuration(self) -> None:
        """Validate runtime model configuration before loading anything."""

        if self._requires_context_examples_for_model(self.standard_model_name):
            raise RuntimeError(
                "MODEL_NAME must point to a standard non-zero-shot model. "
                "Configure zero-shot inference via ZERO_SHOT_MODEL_NAME."
            )

        if self.zero_shot_model_name and not self._requires_context_examples_for_model(
            self.zero_shot_model_name
        ):
            raise RuntimeError(
                "ZERO_SHOT_MODEL_NAME must point to a *_ZS model."
            )

    def _initialize_runtime(self) -> None:
        """Initialize shared device, dtype, and VAD state once."""

        if self.device is not None and self.dtype is not None and self.vad_model is not None:
            return

        with self.runtime_lock:
            if (
                self.device is not None
                and self.dtype is not None
                and self.vad_model is not None
            ):
                return

            device = self._select_device()
            dtype = self._select_dtype(device)
            vad_device = self._select_vad_device(device)
            vad_model = load_silero_vad().to(vad_device)

            self.device = device
            self.dtype = dtype
            self.vad_device = vad_device
            self.vad_model = vad_model

            logger.info(
                "Initialized runtime on %s with dtype=%s (silero_vad=%s)",
                self.device,
                self.dtype,
                self.vad_device,
            )

            self._initialize_word_aligner()
            self._initialize_language_detector()

    def _initialize_language_detector(self) -> None:
        """Initialize the per-chunk LID backend (configured via env)."""

        if self._language_detector_override is not None:
            return  # explicit DI wins

        self.language_detector = load_language_detector(
            OMNILINGUAL_LID_BACKEND,
            model_path=OMNILINGUAL_LID_MODEL_PATH,
            remote_url=OMNILINGUAL_LID_URL,
            remote_token=OMNILINGUAL_LID_TOKEN,
            remote_timeout_seconds=OMNILINGUAL_LID_TIMEOUT_SECONDS,
        )

    def _initialize_word_aligner(self) -> None:
        """Initialize the word aligner (CTC if configured, heuristic otherwise)."""

        if self._word_aligner_override is not None:
            return  # explicit DI wins, no auto-config

        if not self.alignment_model_name:
            logger.info("No ALIGNMENT_MODEL_NAME set; using heuristic word aligner")
            return

        try:
            self.alignment_pipeline = self._load_pipeline(self.alignment_model_name)
            self.word_aligner = CTCForcedAligner(pipeline=self.alignment_pipeline)
            logger.info(
                "Word aligner: CTC forced alignment via %s",
                self.alignment_model_name,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to load alignment model %s; falling back to heuristic aligner",
                self.alignment_model_name,
            )
            self.alignment_pipeline = None

    def _load_pipeline(self, model_name: str) -> ASRInferencePipeline:
        """Load one Omnilingual-ASR pipeline on the shared runtime."""

        if self.device is None or self.dtype is None:
            raise RuntimeError("Runtime not initialized. Call _initialize_runtime() first.")

        logger.info(
            "Loading model %s on %s with dtype=%s...",
            model_name,
            self.device,
            self.dtype,
        )
        pipeline = ASRInferencePipeline(
            model_card=model_name,
            device=self.device,
            dtype=self.dtype,
        )
        logger.info("Model %s loaded successfully on %s", model_name, self.device)
        return pipeline

    def _ensure_standard_pipeline(self) -> ASRInferencePipeline:
        """Ensure the standard pipeline is loaded and ready."""

        if self.standard_pipeline is not None:
            return self.standard_pipeline

        self._initialize_runtime()

        with self.pipeline_lock:
            if self.standard_pipeline is None:
                self.standard_pipeline = self._load_pipeline(self.standard_model_name)

        return self.standard_pipeline

    def _ensure_zero_shot_pipeline(self) -> ASRInferencePipeline:
        """Ensure the zero-shot pipeline is loaded and ready."""

        if not self.zero_shot_model_name:
            raise RuntimeError(
                "Zero-shot model is not configured. Set ZERO_SHOT_MODEL_NAME to enable "
                "the /v1/audio/transcriptions/zero-shot endpoint."
            )

        if self.zero_shot_pipeline is not None:
            return self.zero_shot_pipeline

        self._initialize_runtime()

        with self.pipeline_lock:
            if self.zero_shot_pipeline is None:
                self.zero_shot_pipeline = self._load_pipeline(self.zero_shot_model_name)

        return self.zero_shot_pipeline

    def load_model(self) -> None:
        """Initialize the runtime and eagerly load configured pipelines."""

        self._validate_model_configuration()
        self._ensure_standard_pipeline()

        if OMNILINGUAL_PRELOAD_ZERO_SHOT and self.zero_shot_model_name:
            self._ensure_zero_shot_pipeline()

    def _batch_size_for(
        self,
        chunk_count: int,
        *,
        requires_context_examples: bool,
    ) -> int:
        """Compute a safe effective batch size for the current request."""

        if requires_context_examples:
            return 1

        if OMNILINGUAL_BATCH_SIZE < 1:
            raise RuntimeError("OMNILINGUAL_BATCH_SIZE must be greater than or equal to 1.")

        return min(OMNILINGUAL_BATCH_SIZE, max(1, chunk_count))

    def _decode_context_examples(
        self,
        context_examples: list[UploadedContextExample],
    ) -> list[ContextExample]:
        """Decode uploaded context examples into Omnilingual-ASR context inputs."""

        decoded_examples: list[ContextExample] = []

        for index, example in enumerate(context_examples, start=1):
            text = example.text.strip()
            if not text:
                raise RuntimeError(
                    f"Context example {index} has an empty transcription."
                )

            try:
                decoded_audio = decode_audio_bytes(
                    example.audio_bytes,
                    content_type=example.content_type,
                    filename=example.filename,
                )
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"Context audio decode failed for example {index}."
                ) from e

            context_chunks = self._split_context_audio(decoded_audio)
            context_texts = self._split_context_transcript(
                text,
                context_chunks=context_chunks,
            )

            if len(context_chunks) > 1:
                logger.warning(
                    "Context example %s was %.2fs and has been split into %s fixed "
                    "%.0fs chunk(s). Transcript alignment is heuristic.",
                    index,
                    decoded_audio.duration_seconds,
                    len(context_chunks),
                    MODEL_MAX_AUDIO_SECONDS,
                )

            for context_chunk, context_text in zip(
                context_chunks,
                context_texts,
                strict=True,
            ):
                decoded_examples.append(
                    ContextExample(
                        audio=decoded_audio.to_model_input(context_chunk),
                        text=context_text,
                    )
                )

        return decoded_examples

    def _split_context_audio(self, decoded_audio) -> list[AudioChunk]:
        """Split long zero-shot context audio into fixed windows under the model limit."""

        max_context_samples = int(MODEL_MAX_AUDIO_SECONDS * decoded_audio.sample_rate)
        total_samples = len(decoded_audio.waveform)

        if total_samples <= max_context_samples:
            return [AudioChunk(0, total_samples)]

        chunks: list[AudioChunk] = []
        start_sample = 0
        while start_sample < total_samples:
            end_sample = min(start_sample + max_context_samples, total_samples)
            chunks.append(AudioChunk(start_sample, end_sample))
            start_sample = end_sample

        return chunks

    def _split_context_transcript(
        self,
        text: str,
        *,
        context_chunks: list[AudioChunk],
    ) -> list[str]:
        """Split a transcript across fixed context windows using duration-weighted spans."""

        if len(context_chunks) == 1:
            return [text]

        durations = [chunk.end_sample - chunk.start_sample for chunk in context_chunks]
        words = text.split()
        if len(words) >= len(context_chunks):
            return self._split_sequence_by_weights(words, durations, separator=" ")

        characters = list(text)
        if len(characters) >= len(context_chunks):
            return self._split_sequence_by_weights(characters, durations, separator="")

        logger.warning(
            "Context transcript is too short to split across %s chunk(s); repeating it "
            "for each chunk.",
            len(context_chunks),
        )
        return [text] * len(context_chunks)

    def _split_sequence_by_weights(
        self,
        units: list[str],
        weights: list[int],
        *,
        separator: str,
    ) -> list[str]:
        """Split contiguous text units into non-empty weighted groups."""

        total_units = len(units)
        total_weight = sum(weights)
        remaining_units = total_units
        remaining_weight = total_weight
        start_index = 0
        grouped: list[str] = []

        for index, weight in enumerate(weights):
            segments_left = len(weights) - index
            if segments_left == 1:
                end_index = total_units
            else:
                proportional_count = round((weight / remaining_weight) * remaining_units)
                unit_count = max(1, proportional_count)
                unit_count = min(unit_count, remaining_units - (segments_left - 1))
                end_index = start_index + unit_count

            grouped.append(separator.join(units[start_index:end_index]).strip())

            consumed = end_index - start_index
            start_index = end_index
            remaining_units -= consumed
            remaining_weight -= weight

        return grouped

    async def _transcribe_with_pipeline(
        self,
        *,
        pipeline: ASRInferencePipeline,
        model_name: str,
        audio_bytes: bytes,
        language: str | None = None,
        content_type: str | None = None,
        filename: str | None = None,
        context_examples: list[UploadedContextExample] | None = None,
        chunking_config: ChunkingConfig | None = None,
    ) -> DetailedTranscription:
        """Decode, chunk, and transcribe audio against a specific pipeline."""

        if self.vad_model is None:
            logger.error("Transcription attempted before Silero VAD was loaded")
            raise RuntimeError("Silero VAD not loaded. Call load_model() first.")

        requires_context_examples = self._requires_context_examples_for_model(model_name)
        lang_param = None
        if language and self._is_llm_model_name(model_name) and not requires_context_examples:
            lang_param = map_whisper_to_omnilingual(language)
            logger.debug("Language mapped: %s -> %s", language, lang_param)

        audio_size_kb = len(audio_bytes) / 1024
        logger.info(
            "Starting transcription with %s: %.1fKB, language=%s",
            model_name,
            audio_size_kb,
            lang_param or "auto",
        )

        try:
            decoded_audio = decode_audio_bytes(
                audio_bytes,
                content_type=content_type,
                filename=filename,
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("Audio decode failed.") from e

        with self.vad_lock:
            chunks = build_speech_chunks(
                decoded_audio,
                vad_model=self.vad_model,
                vad_device=self.vad_device,
                config=chunking_config or ChunkingConfig.from_env(),
            )

        chunk_inputs = [decoded_audio.to_model_input(chunk) for chunk in chunks]
        batch_size = self._batch_size_for(
            len(chunk_inputs),
            requires_context_examples=requires_context_examples,
        )

        logger.info(
            "Prepared %s chunk(s) from %.2fs of audio for %s; batch_size=%s",
            len(chunk_inputs),
            decoded_audio.duration_seconds,
            model_name,
            batch_size,
        )

        if requires_context_examples:
            if not context_examples:
                raise RuntimeError(
                    "Zero-shot transcription requires at least one context example. "
                    "Provide matching `context_files` and `context_texts` form fields."
                )

            decoded_context_examples = self._decode_context_examples(context_examples)
            per_chunk_context_examples = [
                list(decoded_context_examples) for _ in chunk_inputs
            ]
            transcriptions = pipeline.transcribe_with_context(
                chunk_inputs,
                context_examples=per_chunk_context_examples,
                batch_size=batch_size,
            )
        elif lang_param:
            transcriptions = pipeline.transcribe(
                chunk_inputs,
                lang=[lang_param] * len(chunk_inputs),
                batch_size=batch_size,
            )
        else:
            transcriptions = pipeline.transcribe(
                chunk_inputs,
                batch_size=batch_size,
            )

        annotations = [
            ChunkAnnotation(chunk=chunk, text=text.strip())
            for chunk, text in zip(chunks, transcriptions, strict=True)
            if text.strip()
        ]

        self._detect_languages(
            annotations,
            waveform=decoded_audio.waveform,
            sample_rate=decoded_audio.sample_rate,
        )
        smooth_chunk_languages(
            annotations,
            config=replace(self.smoothing_config, fallback_language=language),
            sample_rate=decoded_audio.sample_rate,
        )
        self._align_words(
            annotations,
            waveform=decoded_audio.waveform,
            sample_rate=decoded_audio.sample_rate,
        )

        segments = [
            TimedTranscriptSegment(
                start_seconds=ann.start_seconds(decoded_audio.sample_rate),
                end_seconds=ann.end_seconds(decoded_audio.sample_rate),
                text=ann.text,
                words=ann.words,
                language=ann.language,
            )
            for ann in annotations
        ]
        words = [word for ann in annotations for word in ann.words]
        result = join_transcript_texts([ann.text for ann in annotations])
        logger.info(
            "Transcription complete with %s: %s chars from %s chunk(s)",
            model_name,
            len(result),
            len(chunk_inputs),
        )
        return DetailedTranscription(
            text=result,
            duration_seconds=decoded_audio.duration_seconds,
            language=lang_param or language,
            segments=segments,
            words=words,
        )

    def _detect_languages(
        self,
        annotations: list[ChunkAnnotation],
        *,
        waveform,
        sample_rate: int,
    ) -> None:
        """Run direct LID on each chunk; populate language/confidence in-place."""

        for ann in annotations:
            audio_slice = waveform[ann.chunk.start_sample : ann.chunk.end_sample]
            score = self.language_detector.detect(
                text=ann.text,
                audio=audio_slice,
                sample_rate=sample_rate,
            )
            if score is not None:
                ann.language = score.language
                ann.language_confidence = score.confidence

    def _align_words(
        self,
        annotations: list[ChunkAnnotation],
        *,
        waveform,
        sample_rate: int,
    ) -> None:
        """Compute word timestamps for each chunk and propagate the chunk language."""

        for ann in annotations:
            audio_slice = waveform[ann.chunk.start_sample : ann.chunk.end_sample]
            aligned = self.word_aligner.align(
                text=ann.text,
                waveform=audio_slice,
                sample_rate=sample_rate,
                chunk_start_seconds=ann.start_seconds(sample_rate),
                chunk_end_seconds=ann.end_seconds(sample_rate),
            )
            if ann.language is not None:
                aligned = [
                    WordTimestamp(
                        word=w.word,
                        start_seconds=w.start_seconds,
                        end_seconds=w.end_seconds,
                        language=ann.language,
                    )
                    for w in aligned
                ]
            ann.words = aligned

    async def transcribe(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        *,
        content_type: str | None = None,
        filename: str | None = None,
        context_examples: list[UploadedContextExample] | None = None,
        chunking_config: ChunkingConfig | None = None,
    ) -> str:
        """Transcribe audio with the configured standard model."""

        result = await self.transcribe_detailed(
            audio_bytes,
            language=language,
            content_type=content_type,
            filename=filename,
            context_examples=context_examples,
            chunking_config=chunking_config,
        )
        return result.text

    async def transcribe_detailed(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        *,
        content_type: str | None = None,
        filename: str | None = None,
        context_examples: list[UploadedContextExample] | None = None,
        chunking_config: ChunkingConfig | None = None,
    ) -> DetailedTranscription:
        """Transcribe audio with timing metadata for verbose response modes."""

        pipeline = self._ensure_standard_pipeline()
        return await self._transcribe_with_pipeline(
            pipeline=pipeline,
            model_name=self.standard_model_name,
            audio_bytes=audio_bytes,
            language=language,
            content_type=content_type,
            filename=filename,
            context_examples=context_examples,
            chunking_config=chunking_config,
        )

    async def transcribe_zero_shot(
        self,
        audio_bytes: bytes,
        *,
        content_type: str | None = None,
        filename: str | None = None,
        context_examples: list[UploadedContextExample],
        chunking_config: ChunkingConfig | None = None,
    ) -> str:
        """Transcribe audio with the configured zero-shot model."""

        result = await self.transcribe_zero_shot_detailed(
            audio_bytes,
            content_type=content_type,
            filename=filename,
            context_examples=context_examples,
            chunking_config=chunking_config,
        )
        return result.text

    async def transcribe_zero_shot_detailed(
        self,
        audio_bytes: bytes,
        *,
        content_type: str | None = None,
        filename: str | None = None,
        context_examples: list[UploadedContextExample],
        chunking_config: ChunkingConfig | None = None,
    ) -> DetailedTranscription:
        """Transcribe zero-shot audio with timing metadata for verbose modes."""

        pipeline = self._ensure_zero_shot_pipeline()
        return await self._transcribe_with_pipeline(
            pipeline=pipeline,
            model_name=self.zero_shot_model_name,
            audio_bytes=audio_bytes,
            content_type=content_type,
            filename=filename,
            context_examples=context_examples,
            chunking_config=chunking_config,
        )


# Global service instance
asr_service = OmnilingualASRService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler - load models on startup."""

    asr_service.load_model()
    yield
