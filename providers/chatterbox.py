"""Chatterbox and Chatterbox-Turbo provider (optional dependency).

Install ``chatterbox-tts`` in an environment compatible with its pinned
torch/numpy versions. This module is skipped by the registry when the package
is not installed.
"""

import logging
import shutil
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS

from tts_core.audio import AudioChunk
from tts_core.provider import SynthesisModel, TTSProvider, VoiceState

logger = logging.getLogger(__name__)

MODELS = ("chatterbox", "chatterbox-turbo")
ALIASES = {
    "english": "chatterbox",
    "default": "chatterbox",
    "turbo": "chatterbox-turbo",
}

BASE_OPTIONS: dict[str, Any] = {
    "device": "auto",
    "repetition_penalty": 1.2,
    "top_p": 1.0,
    "temperature": 0.8,
    "reference_text": "",
}
TTS_DEFAULTS = {
    **BASE_OPTIONS,
    "min_p": 0.05,
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "top_k": 1000,
    "norm_loudness": True,
}
TURBO_DEFAULTS = {
    **BASE_OPTIONS,
    "top_p": 0.95,
    "min_p": 0.0,
    "exaggeration": 0.0,
    "cfg_weight": 0.0,
    "top_k": 1000,
    "norm_loudness": True,
}
ALLOWED_OPTIONS = frozenset(TTS_DEFAULTS)


@dataclass(frozen=True)
class ChatterboxVoice:
    path: Path | None


class ChatterboxModel(SynthesisModel):
    def __init__(self, model, variant: str, options: Mapping[str, Any]):
        self._model = model
        self._variant = variant
        self._options = dict(options)

    @property
    def sample_rate(self) -> int:
        return int(self._model.sr)

    def load_voice(self, source: str | Path | None) -> VoiceState:
        return ChatterboxVoice(Path(source) if source is not None else None)

    def synthesize(
        self,
        voice: ChatterboxVoice,
        text: str,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[AudioChunk]:
        opts = {**self._options, **(options or {})}
        prompt = str(voice.path) if voice.path is not None else None

        if self._variant == "chatterbox-turbo":
            kwargs = {
                "repetition_penalty": float(opts["repetition_penalty"]),
                "top_p": float(opts["top_p"]),
                "temperature": float(opts["temperature"]),
                "top_k": int(opts["top_k"]),
                "norm_loudness": bool(opts["norm_loudness"]),
            }
        else:
            kwargs = {
                "repetition_penalty": float(opts["repetition_penalty"]),
                "min_p": float(opts["min_p"]),
                "top_p": float(opts["top_p"]),
                "exaggeration": float(opts["exaggeration"]),
                "cfg_weight": float(opts["cfg_weight"]),
                "temperature": float(opts["temperature"]),
            }

        with torch.inference_mode():
            wav = self._model.generate(text, audio_prompt_path=prompt, **kwargs)
        audio = wav.detach().to(device="cpu", dtype=torch.float32).numpy().reshape(-1)
        yield audio

    def export_voice(self, voice: ChatterboxVoice, destination: Path) -> None:
        if voice.path is None:
            raise ValueError("The default Chatterbox voice has no reference file")
        shutil.copyfile(voice.path, destination)


class ChatterboxProvider(TTSProvider):
    """Chatterbox: zero-shot cloning; Turbo is the 350M low-latency variant."""

    name: ClassVar[str] = "chatterbox"
    available_models: ClassVar[tuple[str, ...]] = MODELS
    voice_extension: ClassVar[str] = "wav"
    default_voice: ClassVar[str | None] = "default"
    default_voice_source: ClassVar[str | None] = None

    def resolve_options(self, model: str, options: Mapping[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(options) - ALLOWED_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown chatterbox options: {', '.join(unknown)}")
        variant = ALIASES.get(model, model)
        if variant not in MODELS:
            raise ValueError(f"Unknown Chatterbox model: {model}")
        defaults = TURBO_DEFAULTS if variant == "chatterbox-turbo" else TTS_DEFAULTS
        resolved = {"variant": variant, **defaults, **options}
        resolved["device"] = _resolve_device(str(resolved["device"]))
        return resolved

    def model_cache_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        return {key: options[key] for key in ("variant", "device", "reference_text")}

    def load_model(self, options: Mapping[str, Any]) -> SynthesisModel:
        device = str(options["device"])
        logger.info("Loading Chatterbox (%s) on %s", options["variant"], device)
        if options["variant"] == "chatterbox-turbo":
            model = ChatterboxTurboTTS.from_pretrained(device=device)
        else:
            model = ChatterboxTTS.from_pretrained(device=device)
        return ChatterboxModel(model, str(options["variant"]), options)


def _resolve_device(name: str) -> str:
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")
    if name.startswith("xpu"):
        raise ValueError("XPU is not supported; use a CUDA/MPS/CPU environment")
    return name
