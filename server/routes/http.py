from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from engine import AudioFormat, TTSEngine

router = APIRouter()

MEDIA_TYPES = {
    AudioFormat.OPUS_24K: "audio/ogg; codecs=opus",
    AudioFormat.OPUS_32K: "audio/ogg; codecs=opus",
    AudioFormat.PCM16: "application/octet-stream",
}

class SpeechRequest(BaseModel):
    model: str = "english"
    input: str = Field(min_length=1, max_length=4096)
    voice: str = "alba"
    response_format: AudioFormat = AudioFormat.OPUS_32K
    stream_format: Literal["sse", "audio"] = "audio"

@router.post("/v1/audio/speech")
def speech(body: SpeechRequest, request: Request):
    engine: TTSEngine = request.app.state.engine

    chunks = engine.generate_for(
        language=body.model,
        voice_name=body.voice,
        text=body.input,
        audio_format=body.response_format,
    )

    if body.stream_format == "audio":
        return StreamingResponse(
            chunks,
            media_type=MEDIA_TYPES[body.response_format]
        )

    # SSE support is added separately; this remains raw audio for now.
    return StreamingResponse(chunks, media_type=MEDIA_TYPES[body.response_format])
