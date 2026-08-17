import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from engine import TTSEngine
from server.routes.http import SpeechRequest

router = APIRouter()

_END = object()


def next_chunk(chunks):
    return next(chunks, _END)


@router.websocket("/v1/audio/speech/stream")
async def speech_stream(websocket: WebSocket):
    await websocket.accept()

    engine: TTSEngine = websocket.app.state.engine

    try:
        while True:
            message = await websocket.receive_json()
            body = SpeechRequest.model_validate(message)

            print("body", body)

            chunks = engine.generate_for(
                language=body.model,
                voice_name=body.voice,
                text=body.input,
                audio_format=body.response_format,
            )

            await websocket.send_json({
                "type": "speech.start",
                "format": body.response_format,
                "sample_rate": engine.model.sample_rate,
                "channels": 1,
            })

            while True:
                # Generating one chunk blocks while Pocket TTS works.
                chunk = await asyncio.to_thread(next_chunk, chunks)

                if chunk is _END:
                    break

                await websocket.send_bytes(chunk)

            await websocket.send_json({
                "type": "speech.done",
            })
            print("done")

    except WebSocketDisconnect:
        pass
    except ValidationError as error:
        await websocket.send_json({
            "type": "error",
            "message": error.errors(),
        })
