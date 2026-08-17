from fastapi import FastAPI

from engine import TTSEngine
from server.routes.http import router as http_router


async def health() -> dict[str, str]:
    return {"status": "ok"}


def create_http_server(engine: TTSEngine) -> FastAPI:
    app = FastAPI()
    app.state.engine = engine

    app.add_api_route("/health", health, methods=["GET"])
    app.include_router(http_router)
    return app
