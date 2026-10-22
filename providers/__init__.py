"""Officially supported TTS providers.

To add a backend: implement ``tts_core.provider.TTSProvider`` (and a
``SynthesisModel``), drop it beside ``pocket_tts.py``, and add it to
``_BUILTINS``. External packages can instead publish an entry point in the
``tts_core.providers`` group.

Each provider is imported lazily so a missing provider dependency (for
example, no ``pocket_tts`` install in a GPU-only environment) only removes
that provider instead of all of them.
"""

import importlib
import logging

from tts_core.registry import ProviderRegistry

logger = logging.getLogger(__name__)

# OpenVINO providers are imported before torch-dependent ones so ov.Core()
# initializes while the Intel GPU is still visible.
_BUILTINS = (
    ("providers.qwen3_tts", "Qwen3TTSProvider"),
    ("providers.chatterbox_gguf", "ChatterboxGGUFProvider"),
    ("providers.neutts", "NeutTSProvider"),
    ("providers.chatterbox_onnx", "ChatterboxONNXProvider"),
    ("providers.fish_s1_mini", "FishS1MiniProvider"),
    ("providers.pocket_tts", "PocketTTSProvider"),
    ("providers.kokoro", "KokoroProvider"),
    ("providers.chatterbox", "ChatterboxProvider"),
    ("providers.f5_tts", "F5TTSProvider"),
)

__all__ = ["register_builtin_providers"]


def register_builtin_providers(registry: ProviderRegistry) -> None:
    for module_name, class_name in _BUILTINS:
        try:
            module = importlib.import_module(module_name)
            provider = getattr(module, class_name)()
        except ImportError:
            logger.debug(
                "Skipping TTS provider %s; dependency missing",
                module_name,
                exc_info=True,
            )
            continue
        registry.register(provider, replace=True)
