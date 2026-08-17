import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from engine import TTSEngine
from server.routes.http import router as http_router
from server.routes.websocket import router as websocket_router


async def health() -> dict[str, str]:
    return {"status": "ok"}


def create_http_server(engine: TTSEngine) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.engine = engine
    app.state.engine_lock = asyncio.Lock()

    app.add_api_route("/health", health, methods=["GET"])
    app.include_router(http_router)
    app.include_router(websocket_router)
    return app
