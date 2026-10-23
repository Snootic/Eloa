"""Pocket TTS provider — the built-in reference backend."""

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

import torch
from pocket_tts import TTSModel, export_model_state

import torching  # noqa: F401  # tunes torch thread counts on import
from tts_core.audio import AudioChunk
from tts_core.provider import SynthesisModel, TTSProvider, VoiceState

DEFAULT_VOICE_CONDITIONING = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"

POCKET_DEFAULTS: dict[str, Any] = {
    "temp": 0.7,
    "sampler_decode_steps": 2,
    "noise_clamp": None,
    "eos_threshold": -3.0,
    "quantize": False,
}
POCKET_OPTIONS = frozenset({"language", "config", *POCKET_DEFAULTS})

STREAM_FRAMES_AFTER_EOS = 8
FULL_FRAMES_AFTER_EOS = 5


class PocketTTSModel(SynthesisModel):
    """A loaded Pocket TTS model behind the provider-neutral contract."""

    def __init__(self, model: TTSModel):
        model.pad_with_spaces_for_short_inputs = True
        self._model = model

    @property
    def sample_rate(self) -> int:
        return self._model.sample_rate

    def load_voice(
        self, source: str | Path | None, *, name: str | None = None
    ) -> VoiceState:
        if source is None:
            raise ValueError("Pocket TTS requires an audio reference for every voice")
        return self._model.get_state_for_audio_prompt(source)

    def synthesize(
        self,
        voice: VoiceState,
        text: str,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[AudioChunk]:
        with torch.no_grad():
            for chunk in self._model.generate_audio_stream(
                voice, text, frames_after_eos=STREAM_FRAMES_AFTER_EOS, copy_state=True
            ):
                yield self._to_numpy(chunk)

    def synthesize_all(
        self,
        voice: VoiceState,
        text: str,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> AudioChunk:
        with torch.no_grad():
            audio = self._model.generate_audio(
                text_to_generate=text,
                model_state=voice,
                frames_after_eos=FULL_FRAMES_AFTER_EOS,
                copy_state=True,
            )
        return self._to_numpy(audio)

    def export_voice(self, voice: VoiceState, destination: Path) -> None:
        export_model_state(voice, destination)

    @staticmethod
    def _to_numpy(audio: torch.Tensor) -> AudioChunk:
        return audio.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()


def _pocket_models() -> tuple[str, ...]:
    from pocket_tts.utils.config import CONFIGS_DIR

    return tuple(sorted(path.stem for path in CONFIGS_DIR.glob("*.yaml")))


class PocketTTSProvider(TTSProvider):
    """Pocket TTS: language models with audio-prompt voice cloning."""

    name: ClassVar[str] = "pocket-tts"
    available_models: ClassVar[tuple[str, ...]] = _pocket_models()
    voice_extension: ClassVar[str] = "safetensors"
    default_voice: ClassVar[str | None] = "alba"
    default_voice_source: ClassVar[str | None] = DEFAULT_VOICE_CONDITIONING

    def resolve_options(self, model: str, options: Mapping[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(options) - POCKET_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown pocket-tts options: {', '.join(unknown)}")
        return {"language": model, **POCKET_DEFAULTS, **options}

    def load_model(self, options: Mapping[str, Any]) -> SynthesisModel:
        params = dict(options)
        language = params.pop("language", None)
        config = params.pop("config", None)
        return PocketTTSModel(
            TTSModel.load_model(language=language, config=config, **params)
        )
