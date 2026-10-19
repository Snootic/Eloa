import uvicorn

from server.server import create_http_server
from tts_core import TTSEngine

if __name__ == "__main__":
    engine = TTSEngine(
        provider="pocket-tts",
        model="portuguese",
        voice="camilly-portuguese",
    )

    http_server = create_http_server(engine)
    uvicorn.run(http_server, host="0.0.0.0", port=8082)
