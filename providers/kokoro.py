"""Kokoro-82M provider (CPU-first).

Kokoro ships its own named voices instead of reference clips, so the provider
declares them as ``builtin_voices``. The language pipeline is selected from the
voice prefix (``af_*`` English, ``pf_*`` Portuguese, ...).
"""

import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from kokoro import KPipeline

from tts_core.audio import AudioChunk
from tts_core.provider import SynthesisModel, TTSProvider, VoiceState

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000

VOICES = (
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "em_santa",
    "ff_siwis",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "pm_santa",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
)
LANG_BY_PREFIX = {
    "af": "a",
    "am": "a",
    "bf": "b",
    "bm": "b",
    "ef": "e",
    "em": "e",
    "ff": "f",
    "hf": "h",
    "hm": "h",
    "if": "i",
    "im": "i",
    "jf": "j",
    "jm": "j",
    "pf": "p",
    "pm": "p",
    "zf": "z",
    "zm": "z",
}
LANG_CODES = frozenset("abefhijpz")

DEFAULT_OPTIONS: dict[str, Any] = {
    "lang_code": "a",
    "speed": 1.0,
    "threads": None,
    "repo_id": "hexgrad/Kokoro-82M",
    "device": "cpu",
}
ALLOWED_OPTIONS = frozenset(DEFAULT_OPTIONS)


class KokoroModel(SynthesisModel):
    def __init__(self, pipeline: KPipeline, options: Mapping[str, Any]):
        self._pipelines: dict[str, KPipeline] = {str(options["lang_code"]): pipeline}
        self._options = dict(options)
        self._repo_id = str(options["repo_id"])
        self._device = str(options["device"])

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def load_voice(
        self, source: str | Path | None, *, name: str | None = None
    ) -> VoiceState:
        selected = name or "af_heart"
        if selected not in VOICES:
            raise ValueError(f"Unknown Kokoro voice: {selected}")
        return selected

    def _pipeline_for(self, voice: str) -> KPipeline:
        code = LANG_BY_PREFIX.get(voice[:2], str(self._options["lang_code"]))
        pipeline = self._pipelines.get(code)
        if pipeline is None:
            logger.info("Loading Kokoro pipeline lang_code=%s", code)
            pipeline = KPipeline(
                lang_code=code,
                repo_id=self._repo_id,
                device=self._device,
            )
            self._pipelines[code] = pipeline
        return pipeline

    def synthesize(
        self,
        voice: str,
        text: str,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[AudioChunk]:
        opts = {**self._options, **(options or {})}
        pipeline = self._pipeline_for(voice)
        speed = float(opts["speed"])
        with torch.inference_mode():
            for result in pipeline(text, voice=voice, speed=speed):
                audio = result.audio
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().to(device="cpu", dtype=torch.float32).numpy()
                samples = np.asarray(audio, dtype=np.float32).reshape(-1)
                if samples.size:
                    yield samples

    def export_voice(self, voice: str, destination: Path) -> None:
        raise NotImplementedError("Kokoro voices are built in and cannot be exported")


class KokoroProvider(TTSProvider):
    """Kokoro-82M: fast CPU TTS with 54 built-in voices."""

    name: ClassVar[str] = "kokoro"
    available_models: ClassVar[tuple[str, ...]] = ("kokoro-82m",)
    builtin_voices: ClassVar[tuple[str, ...]] = VOICES
    voice_extension: ClassVar[str] = "pt"
    default_voice: ClassVar[str | None] = "af_heart"
    default_voice_source: ClassVar[str | None] = None

    def resolve_options(self, model: str, options: Mapping[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(options) - ALLOWED_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown kokoro options: {', '.join(unknown)}")
        resolved = {**DEFAULT_OPTIONS, **options}
        lang_code = str(resolved["lang_code"])
        if lang_code not in LANG_CODES:
            raise ValueError(f"Unsupported Kokoro lang_code: {lang_code}")
        resolved["lang_code"] = lang_code
        resolved["device"] = str(resolved["device"]).lower()
        if resolved["device"] not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be cpu, cuda, or mps")
        if resolved["device"] == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        if resolved["device"] == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is not available")
        if not 0.1 <= float(resolved["speed"]) <= 3.0:
            raise ValueError("speed must be between 0.1 and 3.0")
        if resolved["threads"] is not None and int(resolved["threads"]) < 1:
            raise ValueError("threads must be a positive integer or null")
        return resolved

    def model_cache_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            key: options[key] for key in ("lang_code", "device", "threads", "repo_id")
        }

    def load_model(self, options: Mapping[str, Any]) -> SynthesisModel:
        if options["threads"] is not None:
            torch.set_num_threads(int(options["threads"]))
        logger.info(
            "Loading Kokoro-82M lang_code=%s on %s",
            options["lang_code"],
            options["device"],
        )
        pipeline = KPipeline(
            lang_code=str(options["lang_code"]),
            repo_id=str(options["repo_id"]),
            device=str(options["device"]),
        )
        return KokoroModel(pipeline, options)
