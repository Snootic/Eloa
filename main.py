import uvicorn

from engine import TTSEngine
from server.server import create_http_server

if __name__ == "__main__":
    engine = TTSEngine(
        language="portuguese", voice_name="camilly-portuguese", voice_path=None
    )

    http_server = create_http_server(engine)
    uvicorn.run(http_server, host="0.0.0.0", port=8082)
