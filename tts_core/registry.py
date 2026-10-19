"""Provider lookup: built-ins from the app plus installed entry-point plugins."""

import importlib.metadata
import logging
from collections.abc import Iterable, Iterator

from tts_core.provider import TTSProvider

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "tts_core.providers"


class UnknownProviderError(LookupError):
    """Raised when a request names a provider that is not registered."""

    def __init__(self, name: str, available: Iterable[str]):
        self.name = name
        self.available = sorted(available)
        options = ", ".join(self.available) or "none"
        super().__init__(f"Unknown TTS provider {name!r}; available: {options}")


class ProviderRegistry:
    """Name-indexed collection of TTS providers."""

    def __init__(self, providers: Iterable[TTSProvider] = ()):
        self._providers: dict[str, TTSProvider] = {}
        for provider in providers:
            self.register(provider)

    def register(self, provider: TTSProvider, *, replace: bool = False) -> None:
        name = getattr(provider, "name", None)
        if not name:
            raise TypeError("TTS providers must define a non-empty `name`")
        if not getattr(provider, "voice_extension", None):
            raise TypeError(f"Provider {name!r} must define `voice_extension`")
        if not replace and name in self._providers:
            raise ValueError(f"TTS provider {name!r} is already registered")
        self._providers[name] = provider

    def get(self, name: str) -> TTSProvider:
        try:
            return self._providers[name]
        except KeyError:
            raise UnknownProviderError(name, self._providers) from None

    def names(self) -> list[str]:
        return sorted(self._providers)

    def __contains__(self, name: object) -> bool:
        return name in self._providers

    def __iter__(self) -> Iterator[TTSProvider]:
        return iter(self._providers.values())

    def __len__(self) -> int:
        return len(self._providers)


_default_registry: ProviderRegistry | None = None


def default_registry() -> ProviderRegistry:
    """Process-wide registry, populated once from app built-ins and plugins."""
    global _default_registry
    if _default_registry is None:
        registry = ProviderRegistry()
        _load_entry_points(registry)
        _load_builtin_providers(registry)
        _default_registry = registry
    return _default_registry


def _load_entry_points(registry: ProviderRegistry) -> None:
    for entry_point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        try:
            loaded = entry_point.load()
            provider = loaded() if isinstance(loaded, type) else loaded
            registry.register(provider, replace=True)
        except Exception:
            logger.exception("Failed to load TTS provider %r", entry_point.name)


def _load_builtin_providers(registry: ProviderRegistry) -> None:
    try:
        from providers import register_builtin_providers
    except ImportError:
        logger.debug("No built-in TTS providers available", exc_info=True)
        return
    register_builtin_providers(registry)
