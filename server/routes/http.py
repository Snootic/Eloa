from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from engine import TTSEngine
from server.routes.schema import MEDIA_TYPES, SpeechRequest

router = APIRouter()


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
        return StreamingResponse(chunks, media_type=MEDIA_TYPES[body.response_format])

    # SSE support is added separately; this remains raw audio for now.
    return StreamingResponse(chunks, media_type=MEDIA_TYPES[body.response_format])
