"""Model-agnostic TTS engine: model cache, voice cache, and wire encoding."""

import json
import threading
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tts_core.audio import OPUS_SAMPLE_RATE, AudioFormat, to_pcm16
from tts_core.encoding import encode_opus
from tts_core.provider import SynthesisModel, TTSProvider, VoiceState
from tts_core.registry import ProviderRegistry, default_registry

type ModelKey = tuple[str, str]


@dataclass(frozen=True)
class SpeechStream:
    """Encoded audio plus the metadata a client needs to play it."""

    chunks: Iterator[bytes]
    sample_rate: int
    audio_format: AudioFormat


@dataclass(frozen=True)
class ActiveModel:
    """The provider/model pair the engine is currently pointed at."""

    key: ModelKey
    provider: TTSProvider
    model_name: str
    options: Mapping[str, Any]
    synthesis: SynthesisModel


class TTSEngine:
    """Runs any registered provider behind one caching, voice-aware API."""

    def __init__(
        self,
        *,
        provider: str = "pocket-tts",
        model: str = "english",
        options: Mapping[str, Any] | None = None,
        voice: str | None = None,
        voices_dir: str | Path = "voices",
        default_voice_source: str | Path | None = None,
        registry: ProviderRegistry | None = None,
        max_cached_models: int = 2,
    ):
        """Load ``provider``/``model``/``voice`` and make them the active pair.

        ``default_voice_source`` is the conditioning used for any voice name
        that has no export in ``voices_dir``.
        """
        if max_cached_models < 1:
            raise ValueError("max_cached_models must be at least 1")

        self._registry = registry if registry is not None else default_registry()
        self._voices_dir = Path(voices_dir)
        self._default_voice_source = default_voice_source
        self._max_cached_models = max_cached_models
        self._models: OrderedDict[ModelKey, SynthesisModel] = OrderedDict()
        self._voice_caches: dict[ModelKey, dict[str, VoiceState]] = {}
        # Model instances keep per-utterance state (KV caches), so requests on
        # the same instance must not interleave.
        self._generation_locks: dict[ModelKey, threading.Lock] = {}
        self._lock = threading.RLock()
        self._active: ActiveModel | None = None
        self._active_voice: str | None = None

        self._default_provider = provider
        self._default_model = model
        self._default_options = dict(options or {})
        # The voice selected at startup belongs to the startup provider; other
        # providers fall back to their own default voice.
        self._default_voices: dict[str, str] = {}
        active = self._select(provider, model, None)
        self._activate_voice(active, voice)
        if self._active_voice is not None:
            self._default_voices[provider] = self._active_voice

    # ------------------------------------------------------------------
    # Model and voice selection
    # ------------------------------------------------------------------

    def _select(
        self, provider_name: str, model: str, options: Mapping[str, Any] | None
    ) -> ActiveModel:
        provider = self._registry.get(provider_name)
        merged = {**self._default_options, **(options or {})}
        resolved = provider.resolve_options(model, merged)
        key: ModelKey = (
            provider_name,
            json.dumps(
                provider.model_cache_options(resolved), sort_keys=True, default=str
            ),
        )

        with self._lock:
            synthesis = self._models.get(key)
            if synthesis is None:
                synthesis = provider.load_model(resolved)
                self._models[key] = synthesis
                self._voice_caches[key] = {}
                self._generation_locks[key] = threading.Lock()
                while len(self._models) > self._max_cached_models:
                    evicted_key, _ = self._models.popitem(last=False)
                    del self._voice_caches[evicted_key]
                    del self._generation_locks[evicted_key]
            else:
                self._models.move_to_end(key)

            if self._active is None or self._active.key != key:
                self._active_voice = None
            active = ActiveModel(key, provider, model, resolved, synthesis)
            self._active = active
            return active

    def _resolve_voice_source(
        self, active: ActiveModel, voice_name: str, source: str | Path | None
    ) -> str | Path | None:
        path = self._voices_dir / f"{voice_name}.{active.provider.voice_extension}"
        if path.is_file():
            return path
        if source is not None:
            return source
        if self._default_voice_source is not None:
            return self._default_voice_source
        provider = active.provider
        if voice_name == provider.default_voice:
            return provider.default_voice_source
        if provider.voice_exists(voice_name):
            return None
        raise FileNotFoundError(
            f"Voice {voice_name!r} was not found at {path} "
            "and no audio conditioning was supplied"
        )

    def _activate_voice(
        self,
        active: ActiveModel,
        voice_name: str | None,
        source: str | Path | None = None,
    ) -> VoiceState:
        name = (
            voice_name
            or self._default_voices.get(active.provider.name)
            or active.provider.default_voice
        )
        if name is None:
            raise ValueError(
                f"Provider {active.provider.name!r} requires an explicit voice name"
            )

        cache = self._voice_caches[active.key]
        state = cache.get(name)
        if state is None:
            state = active.synthesis.load_voice(
                self._resolve_voice_source(active, name, source), name=name
            )
            cache[name] = state

        self._active_voice = name
        return state

    def load_model(
        self,
        provider: str | None = None,
        model: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> SynthesisModel:
        """Select (loading if needed) a cached model and return it."""
        active = self._select(
            provider or self._default_provider,
            model or self._default_model,
            options,
        )
        return active.synthesis

    def load_voice(
        self,
        voice_name: str | None = None,
        audio_conditioning: str | Path | None = None,
    ) -> VoiceState:
        """Select a voice for the active model, exporting or loading as needed."""
        with self._lock:
            active = self._require_active()
            return self._activate_voice(active, voice_name, audio_conditioning)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_for(
        self,
        *,
        text: str,
        voice: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        options: Mapping[str, Any] | None = None,
        audio_format: AudioFormat = AudioFormat.PCM16,
    ) -> SpeechStream:
        """Select a cached model/voice pair and return a stream bound to it.

        The pair is captured before this call returns, so a later request may
        safely switch the engine's active selection.
        """
        with self._lock:
            active = self._select(
                provider or self._default_provider,
                model or self._default_model,
                options,
            )
            voice_state = self._activate_voice(active, voice)
            sample_rate = (
                active.synthesis.sample_rate
                if audio_format is AudioFormat.PCM16
                else OPUS_SAMPLE_RATE
            )
            return SpeechStream(
                chunks=self._serialized(
                    self._generation_locks[active.key],
                    self._encode(
                        active.synthesis,
                        voice_state,
                        text,
                        audio_format,
                        active.options,
                    ),
                ),
                sample_rate=sample_rate,
                audio_format=audio_format,
            )

    @staticmethod
    def _serialized(lock: threading.Lock, chunks: Iterator[bytes]) -> Iterator[bytes]:
        with lock:
            yield from chunks

    @staticmethod
    def _encode(
        model: SynthesisModel,
        voice: VoiceState,
        text: str,
        audio_format: AudioFormat,
        options: Mapping[str, Any],
    ) -> Iterator[bytes]:
        audio = model.synthesize(voice, text, options=options)
        if audio_format is AudioFormat.PCM16:
            return (to_pcm16(chunk) for chunk in audio)
        return encode_opus(audio, model.sample_rate, audio_format)

    def _active_snapshot(
        self,
    ) -> tuple[threading.Lock, SynthesisModel, VoiceState, Mapping[str, Any]]:
        with self._lock:
            active = self._require_active()
            if self._active_voice is None:
                raise RuntimeError("No voice is selected; call load_voice() first")
            voice = self._voice_caches[active.key][self._active_voice]
            return (
                self._generation_locks[active.key],
                active.synthesis,
                voice,
                active.options,
            )

    def generate_stream(
        self, text: str, audio_format: AudioFormat = AudioFormat.PCM16
    ) -> Iterator[bytes]:
        """Stream encoded audio from the active model/voice."""
        lock, model, voice, options = self._active_snapshot()
        return self._serialized(
            lock, self._encode(model, voice, text, audio_format, options)
        )

    def generate_full(
        self, text: str, audio_format: AudioFormat = AudioFormat.PCM16
    ) -> bytes:
        """Render the whole utterance through the provider's non-streaming path."""
        lock, model, voice, options = self._active_snapshot()
        with lock:
            audio = model.synthesize_all(voice, text, options=options)
        if audio_format is AudioFormat.PCM16:
            return to_pcm16(audio)
        return b"".join(encode_opus([audio], model.sample_rate, audio_format))

    def generate(
        self, text: str, audio_format: AudioFormat = AudioFormat.PCM16
    ) -> bytes:
        """Stream the active model/voice and join the encoded chunks."""
        return b"".join(self.generate_stream(text, audio_format))

    def clone_voice(
        self,
        sample_path: str | Path,
        voice_name: str,
        *,
        provider: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> Path:
        """Export conditioning from an audio sample into ``voices_dir``."""
        with self._lock:
            active = self._select(
                provider or self._default_provider,
                self._default_model,
                options,
            )
            state = active.synthesis.load_voice(sample_path, name=voice_name)
            destination = (
                self._voices_dir / f"{voice_name}.{active.provider.voice_extension}"
            )
            active.synthesis.export_voice(state, destination)
            self._voice_caches[active.key][voice_name] = state
        return destination

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def available_voices(self, provider_name: str) -> list[str]:
        """Built-in and exported voice names for ``provider_name``."""
        provider = self._registry.get(provider_name)
        names = set(provider.builtin_voices)
        if self._voices_dir.is_dir():
            suffix = f".{provider.voice_extension}"
            names.update(
                entry.name.removesuffix(suffix)
                for entry in self._voices_dir.iterdir()
                if entry.is_file() and entry.name.endswith(suffix)
            )
        return sorted(names)

    def _require_active(self) -> ActiveModel:
        if self._active is None:
            raise RuntimeError("No model is selected")
        return self._active

    @property
    def default_provider(self) -> str:
        return self._default_provider

    @property
    def default_model(self) -> str:
        return self._default_model

    def default_voice(self, provider_name: str) -> str | None:
        """Voice a request gets when it names neither a voice nor a source."""
        provider = self._registry.get(provider_name)
        return self._default_voices.get(provider_name) or provider.default_voice

    @property
    def providers(self) -> ProviderRegistry:
        return self._registry

    @property
    def model(self) -> SynthesisModel:
        return self._require_active().synthesis

    @property
    def sample_rate(self) -> int:
        return self._require_active().synthesis.sample_rate

    @property
    def active_voice(self) -> str | None:
        return self._active_voice

    @property
    def voices(self) -> dict[str, VoiceState]:
        active = self._require_active()
        return self._voice_caches[active.key]
