from typing import Any

from pydantic import BaseModel, Field

from tts_core.audio import AudioFormat


class SpeechRequest(BaseModel):
    """Provider-neutral synthesis request.

    ``provider`` and ``model`` fall back to the engine's configured defaults;
    ``options`` are provider-specific overrides (for Pocket TTS: language,
    sampling temperature, decode steps, ...).
    """

    provider: str | None = None
    model: str | None = None
    input: str = Field(min_length=1, max_length=264192)
    voice: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    response_format: AudioFormat = AudioFormat.OPUS_32K
