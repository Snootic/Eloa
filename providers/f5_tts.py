"""F5-TTS provider (optional dependency, non-commercial model weights).

Install ``f5-tts`` in an environment compatible with its torch/torchcodec
stack. This module is skipped by the registry when the package is missing.
"""

import logging
import shutil
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from f5_tts.api import F5TTS

from tts_core.audio import AudioChunk
from tts_core.provider import SynthesisModel, TTSProvider, VoiceState

logger = logging.getLogger(__name__)

MODELS = ("F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base")

DEFAULT_OPTIONS: dict[str, Any] = {
    "device": "auto",
    "nfe_step": 32,
    "cfg_strength": 2.0,
    "sway_sampling_coef": -1.0,
    "speed": 1.0,
    "cross_fade_duration": 0.15,
    "seed": None,
    "reference_text": "",
}
ALLOWED_OPTIONS = frozenset(DEFAULT_OPTIONS)


@dataclass(frozen=True)
class F5Voice:
    path: Path
    text: str


class F5Model(SynthesisModel):
    def __init__(self, model: F5TTS, options: Mapping[str, Any]):
        self._model = model
        self._options = dict(options)

    @property
    def sample_rate(self) -> int:
        return int(self._model.target_sample_rate)

    def load_voice(
        self, source: str | Path | None, *, name: str | None = None
    ) -> VoiceState:
        if source is None:
            raise ValueError(
                "F5-TTS requires a reference clip and transcript for every voice"
            )
        path = Path(source)
        text = str(self._options["reference_text"])
        label = path.with_suffix(".lab")
        if label.is_file():
            text = label.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(
                f"F5-TTS needs the reference transcript: add {label} or set "
                "options.reference_text"
            )
        return F5Voice(path, text)

    def synthesize(
        self,
        voice: F5Voice,
        text: str,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[AudioChunk]:
        opts = {**self._options, **(options or {})}
        wav, _, _ = self._model.infer(
            ref_file=str(voice.path),
            ref_text=voice.text,
            gen_text=text,
            show_info=lambda *args, **kwargs: None,
            progress=None,
            nfe_step=int(opts["nfe_step"]),
            cfg_strength=float(opts["cfg_strength"]),
            sway_sampling_coef=float(opts["sway_sampling_coef"]),
            speed=float(opts["speed"]),
            cross_fade_duration=float(opts["cross_fade_duration"]),
            seed=opts["seed"],
        )
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        yield audio

    def export_voice(self, voice: F5Voice, destination: Path) -> None:
        shutil.copyfile(voice.path, destination)
        destination.with_suffix(".lab").write_text(voice.text, encoding="utf-8")


class F5TTSProvider(TTSProvider):
    """F5-TTS: flow-matching TTS with reference-audio voice cloning."""

    name: ClassVar[str] = "f5-tts"
    available_models: ClassVar[tuple[str, ...]] = MODELS
    voice_extension: ClassVar[str] = "wav"
    default_voice: ClassVar[str | None] = None
    default_voice_source: ClassVar[str | None] = None

    def resolve_options(self, model: str, options: Mapping[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(options) - ALLOWED_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown f5-tts options: {', '.join(unknown)}")
        selected = model if model in MODELS else "F5TTS_v1_Base"
        resolved = {**DEFAULT_OPTIONS, **options, "model": selected}
        resolved["device"] = _resolve_device(str(resolved["device"]))
        if int(resolved["nfe_step"]) < 1:
            raise ValueError("nfe_step must be positive")
        return resolved

    def model_cache_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        return {key: options[key] for key in ("model", "device", "reference_text")}

    def load_model(self, options: Mapping[str, Any]) -> SynthesisModel:
        device = str(options["device"])
        logger.info("Loading F5-TTS (%s) on %s", options["model"], device)
        model = F5TTS(model=str(options["model"]), device=device)
        return F5Model(model, options)


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
