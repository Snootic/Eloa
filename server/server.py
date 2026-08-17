"""
Http and Websocket server for easy-pocket-tts
"""

from fastapi import FastAPI

from engine import TTSEngine
from server.routes.http import router as http_router


def create_http_server(engine: TTSEngine) -> FastAPI:
    app = FastAPI()
    app.state.engine = engine

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.include_router(http_router)
    return app
