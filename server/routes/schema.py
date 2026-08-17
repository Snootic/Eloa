from typing import Literal

from pydantic import BaseModel, Field

from engine import AudioFormat

MEDIA_TYPES = {
    AudioFormat.OPUS_24K: "audio/ogg; codecs=opus",
    AudioFormat.OPUS_32K: "audio/ogg; codecs=opus",
    AudioFormat.PCM16: "application/octet-stream",
}


class SpeechRequest(BaseModel):
    model: str = "english"
    input: str = Field(min_length=1, max_length=264192)
    voice: str = "alba"
    response_format: AudioFormat = AudioFormat.OPUS_32K
    stream_format: Literal["sse", "audio"] = "audio"
