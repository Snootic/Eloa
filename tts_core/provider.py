"""The contract a TTS backend implements to plug into the engine.

A provider owns everything model-specific: which weights to load, what its
model options mean, how conditioning is built, and how audio is generated.
The core owns everything model-agnostic: caching, voice lookup on disk,
locking, and wire encoding.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from tts_core.audio import AudioChunk, as_mono_float32

type ModelOptions = Mapping[str, Any]
type VoiceState = Any


class SynthesisModel(ABC):
    """A loaded model instance that turns text into mono float32 audio."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Sample rate of the audio this model produces."""

    @abstractmethod
    def load_voice(self, source: str | Path | None) -> VoiceState:
        """Build conditioning state from a local file, reference, or ``None``.

        ``None`` is passed only for a provider's ``default_voice`` when that
        provider declares no ``default_voice_source``; providers that cannot
        generate an unconditioned voice should raise ``ValueError``.
        """

    @abstractmethod
    def synthesize(
        self,
        voice: VoiceState,
        text: str,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[AudioChunk]:
        """Yield mono float32 chunks in [-1, 1] as they are produced.

        ``options`` are the request's resolved provider options, which may
        differ from the ones the model was loaded with (sampling parameters).
        """

    def synthesize_all(
        self,
        voice: VoiceState,
        text: str,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> AudioChunk:
        """Generate the whole utterance in one call.

        The default implementation concatenates the streaming API; providers
        with a faster non-streaming path should override it.
        """
        return concatenate(self.synthesize(voice, text, options=options))

    def export_voice(self, voice: VoiceState, destination: Path) -> None:
        """Persist conditioning state so it can be reused without the sample."""
        raise NotImplementedError(f"{type(self).__name__} cannot export voices")


def concatenate(chunks: Iterable[AudioChunk]) -> AudioChunk:
    """Join provider chunks into one mono float32 array."""
    arrays = [as_mono_float32(chunk) for chunk in chunks]
    if not arrays:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(arrays)


class TTSProvider(ABC):
    """An officially supported or third-party TTS backend."""

    name: ClassVar[str]
    """Stable identifier clients use to select this provider."""

    voice_extension: ClassVar[str]
    """Extension of exported voices inside the voices directory."""

    default_voice: ClassVar[str | None] = None
    """Voice used when a request does not name one."""

    default_voice_source: ClassVar[str | None] = None
    """Where ``default_voice`` comes from when it is not on disk."""

    available_models: ClassVar[tuple[str, ...]] = ()
    """Model aliases clients can select from a drop-down."""

    @abstractmethod
    def load_model(self, options: ModelOptions) -> SynthesisModel:
        """Load weights for already-resolved, validated options."""

    def resolve_options(self, model: str, options: Mapping[str, Any]) -> dict[str, Any]:
        """Turn a model alias plus user overrides into full provider options.

        The returned mapping must be JSON-serializable because the engine uses
        it as the model cache key.
        """
        return {"model": model, **options}

    def model_cache_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        """Subset of resolved options that identifies a loaded model.

        Providers with per-request generation options (sampling temperature,
        seeds, ...) override this so those do not trigger reloads. The subset
        must also cover anything voice conditioning depends on.
        """
        return options
