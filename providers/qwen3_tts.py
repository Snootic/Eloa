"""Qwen3-TTS 0.6B/1.7B through qwentts.cpp (GGML, Vulkan/CUDA/CPU).

The model runs in the external ``qwen-tts`` binary; this provider keeps the
engine contract native by invoking it with ``-o -`` and decoding the WAV that
it writes to stdout. Reference voices are converted to 24 kHz with ffmpeg and
encoded once with ``qwen-codec``. When a transcript sidecar is available the
provider uses ICL (``.spk`` + ``.rvq`` + ``.txt``), which preserves the
speaker; without one it falls back to x-vector-only conditioning.
"""

import hashlib
import io
import logging
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import soundfile as sf

from tts_core.audio import AudioChunk
from tts_core.model_catalog import load_model_catalog
from tts_core.provider import SynthesisModel, TTSProvider

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
CACHE_DIR = Path(tempfile.gettempdir()) / "pocket-tts-qwen3"

CATALOG = load_model_catalog().get("qwen3-tts", {})
CATALOG_MODELS = CATALOG.get("models", {})
FALLBACK_MODELS = {
    "qwen3-tts-0.6b": {"size": "0.6b", "quant": "Q8_0"},
    "qwen3-tts-0.6b-q8": {"size": "0.6b", "quant": "Q8_0"},
    "qwen3-tts-0.6b-q6": {"size": "0.6b", "quant": "Q6_K"},
    "qwen3-tts-0.6b-q5": {"size": "0.6b", "quant": "Q5_K_M"},
    "qwen3-tts-0.6b-q4": {"size": "0.6b", "quant": "Q4_K_M"},
    "qwen3-tts-1.7b": {"size": "1.7b", "quant": "Q8_0"},
    "qwen3-tts-1.7b-q8": {"size": "1.7b", "quant": "Q8_0"},
    "qwen3-tts-1.7b-q6": {"size": "1.7b", "quant": "Q6_K"},
    "qwen3-tts-1.7b-q5": {"size": "1.7b", "quant": "Q5_K_M"},
    "qwen3-tts-1.7b-q4": {"size": "1.7b", "quant": "Q4_K_M"},
}
MODEL_ALIASES = {**FALLBACK_MODELS, **CATALOG_MODELS}
TALKER_PATTERN = CATALOG.get("talker_pattern", "qwen-talker-{size}-base-{quant}.gguf")
DEFAULT_CODEC = CATALOG.get("codec", "qwen-tokenizer-12hz-Q8_0.gguf")
QUANT_ALIASES = {
    "q8": "Q8_0",
    "q8_0": "Q8_0",
    "q6": "Q6_K",
    "q6_k": "Q6_K",
    "q5": "Q5_K_M",
    "q5_k_m": "Q5_K_M",
    "q4": "Q4_K_M",
    "q4_k_m": "Q4_K_M",
}

DEFAULT_OPTIONS: dict[str, Any] = {
    "talker": None,
    "models_dir": None,
    "codec": None,
    "binary": None,
    "codec_binary": None,
    "device": "Vulkan0",
    "language": "auto",
    "seed": 42,
    "temperature": None,
    "top_k": None,
    "top_p": None,
    "repetition_penalty": None,
    "max_new_tokens": None,
    "codec_chunk_dur": None,
    "timeout": 300.0,
}
ALLOWED_OPTIONS = frozenset(DEFAULT_OPTIONS)
PER_REQUEST_OPTIONS = frozenset(
    {
        "language",
        "seed",
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "max_new_tokens",
        "codec_chunk_dur",
        "timeout",
    }
)


@dataclass(frozen=True)
class QwenVoiceState:
    """Reference conditioning: speaker embedding plus optional ICL codes."""

    spk: Path
    rvq: Path | None = None
    ref_text: Path | None = None
    source: Path | None = None


def _transcript_for(path: Path) -> Path | None:
    """Find a non-empty .lab/.txt sidecar for a reference clip or voice."""
    for suffix in (".txt", ".lab"):
        candidate = path.with_suffix(suffix)
        if (
            candidate.is_file()
            and candidate.read_text(encoding="utf-8", errors="replace").strip()
        ):
            return candidate
    return None


def _resolve_executable(value: str, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate
    found = shutil.which(value)
    if found is None:
        raise FileNotFoundError(
            f"{label} not found at {value!r} or on PATH; build qwentts.cpp and "
            "pass its path in the provider options"
        )
    return Path(found)


class Qwen3TTSModel(SynthesisModel):
    def __init__(self, options: dict[str, Any]):
        self._options = dict(options)
        self._talker = Path(options["talker"])
        self._codec = Path(options["codec"])
        self._binary = _resolve_executable(str(options["binary"]), "qwen-tts")
        self._codec_binary = _resolve_executable(
            str(options["codec_binary"]), "qwen-codec"
        )
        self._device = str(options["device"])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def _run(
        self, command: list[str], *, text: str | None = None, timeout: float
    ) -> bytes:
        logger.debug("Running %s", " ".join(command))
        try:
            result = subprocess.run(
                command,
                input=text.encode("utf-8") if text is not None else None,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"Qwen3-TTS did not finish within {timeout:.0f}s"
            ) from error
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", "replace").strip().splitlines()
            detail = "\n".join(stderr[-8:]) or "no stderr output"
            raise RuntimeError(
                f"Qwen3-TTS failed with exit code {result.returncode}:\n{detail}"
            )
        return result.stdout

    def _expected_spk_bytes(self) -> int:
        return 8192 if "1.7b" in self._talker.name.lower() else 4096

    @staticmethod
    def _reference_audio_for(spk: Path) -> Path | None:
        for suffix in (".wav", ".mp3", ".flac", ".ogg", ".ref.wav", ".ref.mp3"):
            candidate = spk.with_suffix(suffix)
            if candidate.is_file():
                return candidate
        return None

    def _conditioning_for(self, source: Path) -> QwenVoiceState:
        if source.suffix.lower() == ".spk" and source.is_file():
            if source.stat().st_size != self._expected_spk_bytes():
                reference = self._reference_audio_for(source)
                if reference is None:
                    raise ValueError(
                        f"Speaker embedding {source} was built for a different "
                        "model size; export the voice from its reference audio"
                    )
                return self._conditioning_for(reference)
            rvq = source.with_suffix(".rvq")
            return QwenVoiceState(
                spk=source,
                rvq=rvq if rvq.is_file() else None,
                ref_text=_transcript_for(source),
                source=self._reference_audio_for(source),
            )

        transcript_path = _transcript_for(source)
        transcript = (
            transcript_path.read_text(encoding="utf-8", errors="replace").strip()
            if transcript_path
            else ""
        )
        stat = source.stat()
        key = hashlib.sha256(
            "\0".join(
                (
                    str(source.resolve()),
                    str(stat.st_size),
                    str(stat.st_mtime_ns),
                    str(self._talker),
                    str(self._codec),
                    self._device,
                    transcript,
                )
            ).encode()
        ).hexdigest()
        spk = CACHE_DIR / f"{key}.spk"
        rvq = CACHE_DIR / f"{key}.rvq"
        ref_text = CACHE_DIR / f"{key}.txt"
        if spk.is_file() and (not transcript or (rvq.is_file() and ref_text.is_file())):
            return QwenVoiceState(
                spk,
                rvq if rvq.is_file() else None,
                ref_text if ref_text.is_file() else None,
                source=source,
            )

        wav = CACHE_DIR / f"{key}.wav"
        try:
            converted = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-c:a",
                    "pcm_s16le",
                    str(wav),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )
            if converted.returncode != 0:
                detail = converted.stderr.decode("utf-8", "replace").strip()
                raise RuntimeError(
                    f"Could not decode reference audio {source}: {detail}"
                )
            self._run(
                [
                    str(self._codec_binary),
                    "--model",
                    str(self._codec),
                    "--talker",
                    str(self._talker),
                    "--device",
                    self._device,
                    "-i",
                    str(wav),
                ],
                timeout=float(self._options["timeout"]),
            )
            if not spk.is_file():
                raise RuntimeError(
                    f"qwen-codec did not produce a speaker embedding for {source}"
                )
            if transcript:
                if not rvq.is_file():
                    raise RuntimeError(
                        f"qwen-codec did not produce ICL codes for {source}"
                    )
                ref_text.write_text(transcript + "\n", encoding="utf-8")
            else:
                rvq.unlink(missing_ok=True)
                ref_text.unlink(missing_ok=True)
            return QwenVoiceState(
                spk,
                rvq if rvq.is_file() else None,
                ref_text if ref_text.is_file() else None,
                source=source,
            )
        finally:
            wav.unlink(missing_ok=True)

    def load_voice(
        self, source: str | Path | None, *, name: str | None = None
    ) -> QwenVoiceState:
        if source is None:
            raise ValueError(
                "Qwen3-TTS needs a reference WAV; add a .spk voice under voices/ "
                "or pass audio conditioning"
            )
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Reference audio not found: {path}")
        return self._conditioning_for(path)

    def export_voice(self, voice: QwenVoiceState, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(voice.spk, destination)
        for suffix, source in ((".rvq", voice.rvq), (".txt", voice.ref_text)):
            target = destination.with_suffix(suffix)
            if source is None:
                target.unlink(missing_ok=True)
            else:
                shutil.copyfile(source, target)
        for suffix in (".ref.wav", ".ref.mp3"):
            destination.with_suffix(suffix).unlink(missing_ok=True)
        if voice.source is not None:
            shutil.copyfile(
                voice.source, destination.with_suffix(f".ref{voice.source.suffix}")
            )

    def synthesize(
        self,
        voice: QwenVoiceState,
        text: str,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[AudioChunk]:
        opts = {**self._options, **(options or {})}
        command = [
            str(self._binary),
            "--model",
            str(self._talker),
            "--codec",
            str(self._codec),
            "--device",
            self._device,
            "--ref-spk",
            str(voice.spk),
            "--format",
            "wav16",
            "-o",
            "-",
        ]
        language = str(opts["language"] or "").strip()
        if language and language.lower() != "auto":
            command += ["--lang", language]
        if voice.rvq is not None and voice.ref_text is not None:
            command += [
                "--ref-rvq",
                str(voice.rvq),
                "--ref-text",
                str(voice.ref_text),
            ]
        if int(opts["seed"]) >= 0:
            command += ["--seed", str(int(opts["seed"]))]
        if opts["temperature"] is not None:
            command += ["--temp", str(float(opts["temperature"]))]
        if opts["top_k"] is not None:
            command += ["--top-k", str(int(opts["top_k"]))]
        if opts["top_p"] is not None:
            command += ["--top-p", str(float(opts["top_p"]))]
        if opts["repetition_penalty"] is not None:
            command += ["--rep-pen", str(float(opts["repetition_penalty"]))]
        if opts["max_new_tokens"] is not None:
            command += ["--max-new", str(int(opts["max_new_tokens"]))]
        if opts["codec_chunk_dur"] is not None:
            command += ["--codec-chunk-dur", str(float(opts["codec_chunk_dur"]))]

        payload = self._run(command, text=text, timeout=float(opts["timeout"]))
        if not payload:
            raise RuntimeError("Qwen3-TTS produced no audio")
        samples, _ = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
        audio = np.ascontiguousarray(samples.mean(axis=1), dtype=np.float32)
        if audio.size:
            yield np.clip(audio, -1.0, 1.0)


class Qwen3TTSProvider(TTSProvider):
    """Qwen3-TTS 12 Hz via qwentts.cpp with cached speaker embeddings."""

    name: ClassVar[str] = "qwen3-tts"
    voice_extension: ClassVar[str] = "spk"
    available_models: ClassVar[tuple[str, ...]] = tuple(MODEL_ALIASES)
    default_voice: ClassVar[str | None] = None
    default_voice_source: ClassVar[str | None] = None

    @staticmethod
    def _model_spec(model: str) -> dict[str, str]:
        spec = MODEL_ALIASES.get(model)
        if isinstance(spec, dict) and "size" in spec and "quant" in spec:
            return {"size": str(spec["size"]), "quant": str(spec["quant"])}
        size = "1.7b" if "1.7b" in model else "0.6b"
        suffix = model.rsplit("-", 1)[-1].lower()
        return {"size": size, "quant": QUANT_ALIASES.get(suffix, "Q8_0")}

    def resolve_options(self, model: str, options: Mapping[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(options) - ALLOWED_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown qwen3-tts options: {', '.join(unknown)}")
        resolved = {**DEFAULT_OPTIONS, **options}
        spec = self._model_spec(model)
        models = Path(
            str(resolved["models_dir"] or CATALOG.get("models_dir", "models"))
        )
        if resolved["talker"] is None:
            talker = models / TALKER_PATTERN.format(
                size=spec["size"], quant=spec["quant"]
            )
        else:
            talker = Path(str(resolved["talker"]))
            name = re.sub(r"(0\.6b|1\.7b)", spec["size"], talker.name)
            name = re.sub(
                r"(?:BF16|F32|Q\d+(?:_[A-Za-z0-9]+)*)\.gguf$",
                f"{spec['quant']}.gguf",
                name,
            )
            talker = talker.with_name(name)
        resolved["talker"] = str(talker)
        resolved["models_dir"] = str(models)
        if resolved["codec"] is None:
            resolved["codec"] = str(models / DEFAULT_CODEC)
        if resolved["binary"] is None:
            resolved["binary"] = "build/qwen-tts"
        if resolved["codec_binary"] is None:
            resolved["codec_binary"] = "build/qwen-codec"
        return {"model": model, **resolved}

    def model_cache_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in options.items()
            if key not in PER_REQUEST_OPTIONS
        }

    def load_model(self, options: dict[str, Any]) -> Qwen3TTSModel:
        for key in ("talker", "codec"):
            path = Path(str(options[key]))
            if not path.is_file():
                raise FileNotFoundError(
                    f"Qwen3-TTS {key} file not found: {path}. Download the GGUFs "
                    "from Serveurperso/Qwen3-TTS-GGUF or set the provider option."
                )
        _resolve_executable(str(options["binary"]), "qwen-tts")
        _resolve_executable(str(options["codec_binary"]), "qwen-codec")
        return Qwen3TTSModel(options)
