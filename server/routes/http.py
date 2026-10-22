import json
import re
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from server.routes.schema import SpeechRequest
from tts_core.audio import MEDIA_TYPES
from tts_core.engine import TTSEngine

router = APIRouter()

VOICE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
AUDIO_SUFFIXES = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".oga",
    ".opus",
    ".m4a",
    ".aac",
    ".webm",
}
MAX_REFERENCE_BYTES = 32 * 1024 * 1024


@router.post("/v1/audio/speech")
def speech(body: SpeechRequest, request: Request):
    engine: TTSEngine = request.app.state.engine

    stream = engine.generate_for(
        provider=body.provider,
        model=body.model,
        options=body.options,
        voice=body.voice,
        text=body.input,
        audio_format=body.response_format,
    )

    return StreamingResponse(
        stream.chunks,
        media_type=MEDIA_TYPES[body.response_format],
        headers={
            "X-Sample-Rate": str(stream.sample_rate),
            "X-Audio-Format": str(stream.audio_format),
            "X-TTS-Provider": stream.provider,
            "X-TTS-Model": stream.model,
            "X-TTS-Fallback": "1" if stream.fallback else "0",
        },
    )


@router.get("/v1/audio/providers")
def list_providers(request: Request) -> dict[str, object]:
    engine: TTSEngine = request.app.state.engine

    return {
        "default_provider": engine.default_provider,
        "default_model": engine.default_model,
        "providers": [
            {
                "name": provider.name,
                "models": list(provider.available_models),
                "default_voice": engine.default_voice(provider.name),
                "provider_default_voice": provider.default_voice,
                "voices": engine.available_voices(provider.name),
            }
            for provider in engine.providers
        ],
    }


@router.post("/v1/audio/voices/clone")
async def clone_voice(
    request: Request,
    file: Annotated[UploadFile, File()],
    name: Annotated[str, Form()],
    provider: Annotated[str | None, Form()] = None,
    reference_text: Annotated[str | None, Form()] = None,
    options: Annotated[str | None, Form()] = None,
    overwrite: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    """Encode a reference clip into a reusable voice for one provider.

    The transcript is optional; when supplied it is passed to providers that
    condition on reference text (FishAudio) and persisted next to the voice.
    ``options`` is a JSON object of provider options, matching the speech body.
    """
    engine: TTSEngine = request.app.state.engine

    clone_options: dict[str, object] = {}
    if options:
        try:
            decoded = json.loads(options)
        except json.JSONDecodeError as error:
            raise HTTPException(
                status_code=400, detail=f"options is not valid JSON: {error}"
            ) from error
        if not isinstance(decoded, dict):
            raise HTTPException(status_code=400, detail="options must be an object")
        clone_options = decoded

    if not VOICE_NAME_PATTERN.fullmatch(name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Voice name must be 1-64 characters of letters, digits, '.', "
                "'_' or '-', and start with a letter or digit"
            ),
        )

    provider_name = provider or engine.default_provider
    engine.providers.get(provider_name)  # raises UnknownProviderError -> 404

    if name in engine.available_voices(provider_name) and not overwrite:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Voice {name!r} already exists for {provider_name}; "
                "send overwrite=true to replace it"
            ),
        )

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio file type {suffix or '(none)'!r}",
        )

    with tempfile.TemporaryDirectory(prefix="tts-clone-") as directory:
        source = Path(directory) / f"reference{suffix}"
        size = 0
        with source.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_REFERENCE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Reference audio exceeds "
                            f"{MAX_REFERENCE_BYTES // (1024 * 1024)} MiB"
                        ),
                    )
                output.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if reference_text and reference_text.strip():
            source.with_suffix(".lab").write_text(
                reference_text.strip(), encoding="utf-8"
            )

        try:
            destination = engine.clone_voice(
                source, name, provider=provider_name, options=clone_options
            )
        except NotImplementedError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "provider": provider_name,
        "voice": name,
        "file": destination.name,
        "voices": engine.available_voices(provider_name),
    }
