"""HTTP server entrypoint, configurable through environment variables.

The defaults keep the original Pocket TTS Portuguese setup. Containers and
deployments can select another provider/model/voice without editing code:

  TTS_PROVIDER, TTS_MODEL, TTS_VOICE, TTS_VOICE_SOURCE, TTS_OPTIONS (JSON),
  TTS_VOICES_DIR, TTS_MAX_CACHED_MODELS, TTS_HOST, TTS_PORT.
"""

import json
import os

import uvicorn

from server.server import create_http_server
from tts_core import TTSEngine


def _options() -> dict[str, object]:
    raw = os.getenv("TTS_OPTIONS", "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SystemExit(f"TTS_OPTIONS is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit("TTS_OPTIONS must be a JSON object")
    return value


def main() -> None:
    provider = os.getenv("TTS_PROVIDER", "pocket-tts")
    model = os.getenv("TTS_MODEL", "portuguese")
    voice = os.getenv("TTS_VOICE", "camilly-portuguese") or None
    source = os.getenv("TTS_VOICE_SOURCE") or None

    engine = TTSEngine(
        provider=provider,
        model=model,
        options=_options(),
        voice=voice,
        voices_dir=os.getenv("TTS_VOICES_DIR", "voices"),
        default_voice_source=source,
        max_cached_models=int(os.getenv("TTS_MAX_CACHED_MODELS", "2")),
    )

    host = os.getenv("TTS_HOST", "0.0.0.0")
    port = int(os.getenv("TTS_PORT", "8082"))
    uvicorn.run(
        create_http_server(engine),
        host=host,
        port=port,
        log_level=os.getenv("TTS_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
