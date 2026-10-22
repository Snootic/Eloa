import asyncio
from collections.abc import Iterator

import auth
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from server.routes.schema import SpeechRequest
from tts_core.engine import TTSEngine

router = APIRouter()

_END = object()


def next_chunk(chunks: Iterator[bytes]):
    return next(chunks, _END)


@router.websocket("/v1/audio/speech/stream")
async def speech_stream(websocket: WebSocket):
    try:
        auth.check(
            websocket.headers.get("authorization"),
            websocket.query_params.get("token"),
        )
    except auth.AuthError:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "unauthorized"})
        await websocket.close(code=4401)
        return
    await websocket.accept()

    engine: TTSEngine = websocket.app.state.engine

    try:
        while True:
            message = await websocket.receive_json()
            body = SpeechRequest.model_validate(message)

            stream = engine.generate_for(
                provider=body.provider,
                model=body.model,
                options=body.options,
                voice=body.voice,
                text=body.input,
                audio_format=body.response_format,
            )

            await websocket.send_json(
                {
                    "type": "speech.start",
                    "format": str(stream.audio_format),
                    "sample_rate": stream.sample_rate,
                    "channels": 1,
                    "provider": stream.provider,
                    "model": stream.model,
                    "fallback": stream.fallback,
                }
            )

            while True:
                # Generation is synchronous; keep the event loop free.
                chunk = await asyncio.to_thread(next_chunk, stream.chunks)

                if chunk is _END:
                    break

                await websocket.send_bytes(chunk)

            await websocket.send_json(
                {
                    "type": "speech.done",
                }
            )

    except WebSocketDisconnect:
        pass
    except ValidationError as error:
        await websocket.send_json(
            {
                "type": "error",
                "message": error.errors(),
            }
        )
    except Exception as error:  # noqa: BLE001  # report any provider failure to the client
        await websocket.send_json(
            {
                "type": "error",
                "message": str(error),
            }
        )
