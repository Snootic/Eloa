from fastapi import APIRouter, WebSocket

router = APIRouter()

@router.websocket("/v1/audio/stream")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
