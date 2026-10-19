"""Provider-agnostic text-to-speech core.

The core knows nothing about any specific model. Backends implement
``TTSProvider``/``SynthesisModel`` and register themselves; Pocket TTS ships
as the built-in reference provider in the ``providers`` package.
"""

from tts_core.audio import MEDIA_TYPES, AudioChunk, AudioFormat
from tts_core.engine import SpeechStream, TTSEngine
from tts_core.provider import SynthesisModel, TTSProvider, VoiceState
from tts_core.registry import ProviderRegistry, UnknownProviderError, default_registry

__all__ = [
    "MEDIA_TYPES",
    "AudioChunk",
    "AudioFormat",
    "ProviderRegistry",
    "SpeechStream",
    "SynthesisModel",
    "TTSEngine",
    "TTSProvider",
    "UnknownProviderError",
    "VoiceState",
    "default_registry",
]
