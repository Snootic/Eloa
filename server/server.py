import asyncio

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import auth
from server.routes.http import router as http_router
from server.routes.websocket import router as websocket_router
from tts_core.engine import TTSEngine
from tts_core.registry import UnknownProviderError


async def health() -> dict[str, str]:
    return {"status": "ok"}


def _error_response(
    request: Request, status_code: int, error: Exception
) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": str(error)})


def create_http_server(engine: TTSEngine) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Sample-Rate",
            "X-Audio-Format",
            "X-TTS-Provider",
            "X-TTS-Model",
            "X-TTS-Fallback",
        ],
    )
    app.state.engine = engine
    app.state.engine_lock = asyncio.Lock()

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        if request.url.path.startswith("/v1/"):
            try:
                auth.check(request.headers.get("authorization"))
            except auth.AuthError:
                return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        return await call_next(request)

    async def unknown_provider(request: Request, error: Exception) -> JSONResponse:
        return _error_response(request, 404, error)

    async def bad_request(request: Request, error: Exception) -> JSONResponse:
        return _error_response(request, 400, error)

    async def not_found(request: Request, error: Exception) -> JSONResponse:
        return _error_response(request, 404, error)

    app.add_exception_handler(UnknownProviderError, unknown_provider)
    app.add_exception_handler(ValueError, bad_request)
    app.add_exception_handler(FileNotFoundError, not_found)

    app.add_api_route("/health", health, methods=["GET"])
    app.include_router(http_router)
    app.include_router(websocket_router)
    return app
