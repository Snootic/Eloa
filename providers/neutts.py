"""NeuTTS Air/Nano provider.

NeuTTS is a small LLM + NeuCodec TTS stack with instant voice cloning. The
backbone can be a Transformers checkpoint or a GGUF file (llama-cpp-python);
the codec is loaded through ``neucodec``. Reference audio and its transcript
are both required for cloning.
"""

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from neutts import NeuTTS

from tts_core.audio import AudioChunk
from tts_core.model_catalog import load_model_catalog
from tts_core.provider import SynthesisModel, TTSProvider

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
DEFAULT_OPTIONS: dict[str, Any] = {
    "backbone_repo": None,
    "codec_repo": "neuphonic/neucodec",
    "backbone_device": "cpu",
    "codec_device": "cpu",
    "language": None,
    "seed": None,
    "temperature": 1.0,
    "top_k": 50,
}
ALLOWED_OPTIONS = frozenset(DEFAULT_OPTIONS)
PER_REQUEST_OPTIONS = frozenset({"temperature", "top_k"})
FALLBACK_MODELS = {
    "neutts-air": {"backbone_repo": "neuphonic/neutts-air", "language": "en-us"},
    "neutts-nano": {"backbone_repo": "neuphonic/neutts-nano", "language": "en-us"},
}
NEUTTS_CATALOG = load_model_catalog().get("neutts", {})
MODEL_ALIASES = {**FALLBACK_MODELS, **NEUTTS_CATALOG.get("models", {})}


def _transcript_for(path: Path) -> str | None:
    for suffix in (".txt", ".lab"):
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                return text
    return None


@dataclass(frozen=True)
class NeutVoiceState:
    codes: torch.Tensor
    ref_text: str


class NeutTSModel(SynthesisModel):
    def __init__(self, tts: NeuTTS, options: Mapping[str, Any]):
        self._tts = tts
        self._options = dict(options)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def load_voice(
        self, source: str | Path | None, *, name: str | None = None
    ) -> NeutVoiceState:
        if source is None:
            raise ValueError("NeuTTS needs a reference clip and transcript")
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Reference audio not found: {path}")

        if path.suffix.lower() == ".pt":
            codes = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(codes, dict):
                codes = codes["codes"]
            transcript = _transcript_for(path)
            if transcript is None:
                raise ValueError(
                    f"NeuTTS needs the reference transcript next to {path} "
                    "(<name>.txt or <name>.lab)"
                )
            return NeutVoiceState(torch.as_tensor(codes), transcript)

        transcript = _transcript_for(path)
        if transcript is None:
            raise ValueError(
                "NeuTTS needs the reference transcript; provide reference_text "
                "when cloning or a .txt/.lab sidecar next to the clip"
            )
        codes = self._tts.encode_reference(str(path))
        return NeutVoiceState(torch.as_tensor(codes), transcript)

    def export_voice(self, voice: NeutVoiceState, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"codes": voice.codes}, destination)
        destination.with_suffix(".txt").write_text(
            voice.ref_text + "\n", encoding="utf-8"
        )

    def synthesize(
        self,
        voice: NeutVoiceState,
        text: str,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[AudioChunk]:
        opts = {**self._options, **(options or {})}
        wav = self._tts.infer(
            text,
            voice.codes,
            voice.ref_text,
            temperature=float(opts["temperature"]),
            top_k=int(opts["top_k"]),
        )
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        if audio.size:
            yield np.clip(audio, -1.0, 1.0)


class NeutTSProvider(TTSProvider):
    """NeuTTS Air/Nano with reference-audio voice cloning."""

    name: ClassVar[str] = "neutts"
    voice_extension: ClassVar[str] = "pt"
    available_models: ClassVar[tuple[str, ...]] = tuple(MODEL_ALIASES)
    default_voice: ClassVar[str | None] = None
    default_voice_source: ClassVar[str | None] = None

    def resolve_options(self, model: str, options: Mapping[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(options) - ALLOWED_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown neutts options: {', '.join(unknown)}")
        resolved = {**DEFAULT_OPTIONS, **options}
        spec = MODEL_ALIASES.get(model, FALLBACK_MODELS["neutts-nano"])
        if resolved["backbone_repo"] is None:
            resolved["backbone_repo"] = spec.get("backbone_repo")
        if resolved["language"] is None:
            resolved["language"] = spec.get("language", "en-us")
        return {"model": model, **resolved}

    def model_cache_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            key: value
            for key, value in options.items()
            if key not in PER_REQUEST_OPTIONS
        }

    def load_model(self, options: Mapping[str, Any]) -> NeutTSModel:
        tts = NeuTTS(
            backbone_repo=str(options["backbone_repo"]),
            backbone_device=str(options["backbone_device"]),
            codec_repo=str(options["codec_repo"]),
            codec_device=str(options["codec_device"]),
            language=str(options["language"]),
            seed=options["seed"],
        )
        return NeutTSModel(tts, options)
