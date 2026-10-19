"""Wire formats and sample conversion shared by every TTS provider."""

from enum import StrEnum

import numpy as np

type AudioChunk = np.ndarray  # 1-D float32 mono samples in [-1, 1]

OPUS_SAMPLE_RATE = 48_000  # Ogg Opus always decodes at 48 kHz


class AudioFormat(StrEnum):
    """Wire formats the engine can emit."""

    PCM16 = "pcm16"
    OPUS_24K = "opus_24k"
    OPUS_32K = "opus_32k"

    @property
    def opus_bitrate(self) -> str | None:
        """Target bitrate for Opus formats, or ``None`` for raw PCM."""
        match self:
            case AudioFormat.OPUS_24K:
                return "24k"
            case AudioFormat.OPUS_32K:
                return "32k"
            case _:
                return None


MEDIA_TYPES: dict[AudioFormat, str] = {
    AudioFormat.PCM16: "application/octet-stream",
    AudioFormat.OPUS_24K: "audio/ogg; codecs=opus",
    AudioFormat.OPUS_32K: "audio/ogg; codecs=opus",
}


def as_mono_float32(chunk: AudioChunk) -> AudioChunk:
    """Normalize a provider chunk to mono float32 clipped to [-1, 1]."""
    samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
    return np.clip(samples, -1.0, 1.0)


def to_pcm16(chunk: AudioChunk) -> bytes:
    """Encode float32 samples as little-endian signed 16-bit PCM."""
    return (as_mono_float32(chunk) * 32767).astype("<i2").tobytes()


def to_f32le(chunk: AudioChunk) -> bytes:
    """Encode float32 samples as raw little-endian float32, as ffmpeg expects."""
    return as_mono_float32(chunk).astype("<f4", copy=False).tobytes()
