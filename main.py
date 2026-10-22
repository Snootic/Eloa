"""HTTP server entrypoint, configurable through environment variables.

The defaults keep the original Pocket TTS Portuguese setup. Containers and
deployments can select another provider/model/voice without editing code:

  TTS_PROVIDER, TTS_MODEL, TTS_VOICE, TTS_VOICE_SOURCE, TTS_OPTIONS (JSON),
  TTS_PROVIDER_OPTIONS (JSON map keyed by provider name),
  TTS_PROVIDER_OPTIONS_FILE (shared JSON map, overridden by TTS_PROVIDER_OPTIONS),
  TTS_MODEL_CATALOG (shared model alias catalog),
  TTS_VOICES_DIR, TTS_LAZY_LOAD, TTS_MAX_LOADED_MODELS,
  TTS_FALLBACK_PROVIDER, TTS_FALLBACK_MODEL, TTS_FALLBACK_OPTIONS,
  TTS_HOST, TTS_PORT.
"""

import json
import os
from pathlib import Path

import uvicorn

from server.server import create_http_server
from tts_core import TTSEngine


def _json_object(name: str) -> dict[str, object]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SystemExit(f"{name} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"{name} must be a JSON object")
    return value


def _json_file(name: str) -> dict[str, object]:
    path = os.getenv(name)
    if not path:
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"{name} could not be read as JSON: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"{name} must contain a JSON object")
    return value


def _provider_options() -> dict[str, dict[str, object]]:
    value = _json_file("TTS_PROVIDER_OPTIONS_FILE")
    overrides = _json_object("TTS_PROVIDER_OPTIONS")
    for provider, options in overrides.items():
        if not isinstance(options, dict):
            raise SystemExit("TTS_PROVIDER_OPTIONS values must be JSON objects")
        value.setdefault(provider, {})
        if not isinstance(value[provider], dict):
            raise SystemExit(
                f"TTS_PROVIDER_OPTIONS_FILE entry for {provider!r} must be an object"
            )
        value[provider].update(options)
    if not all(isinstance(item, dict) for item in value.values()):
        raise SystemExit("provider options values must be JSON objects")
    return value


def main() -> None:
    provider = os.getenv("TTS_PROVIDER", "pocket-tts")
    model = os.getenv("TTS_MODEL", "portuguese")
    voice = os.getenv("TTS_VOICE", "eloa") or None
    source = os.getenv("TTS_VOICE_SOURCE") or None
    lazy = os.getenv("TTS_LAZY_LOAD", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    max_loaded_models = int(
        os.getenv(
            "TTS_MAX_LOADED_MODELS",
            os.getenv("TTS_MAX_CACHED_MODELS", "1"),
        )
    )
    fallback_provider = os.getenv("TTS_FALLBACK_PROVIDER") or None
    fallback_model = os.getenv("TTS_FALLBACK_MODEL") or None

    engine = TTSEngine(
        provider=provider,
        model=model,
        options=_json_object("TTS_OPTIONS"),
        provider_options=_provider_options(),
        voice=voice,
        voices_dir=os.getenv("TTS_VOICES_DIR", "voices"),
        default_voice_source=source,
        max_loaded_models=max_loaded_models,
        lazy=lazy,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
        fallback_options=_json_object("TTS_FALLBACK_OPTIONS"),
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
