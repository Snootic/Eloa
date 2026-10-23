"""Qwen3-TTS 0.6B/1.7B through qwentts.cpp (GGML, Vulkan/CUDA/CPU).

The native backend loads ``libqwen.so`` through ctypes and keeps the model
resident across requests when ``native_library`` is configured. The
``qwen-tts`` subprocess remains available with ``backend: "cli"``; reference
voices are converted to 24 kHz with ffmpeg and encoded once with
``qwen-codec``. When a transcript sidecar is available the provider uses ICL
(``.spk`` + ``.rvq`` + ``.txt``), which preserves the speaker; without one it
falls back to x-vector-only conditioning.
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
    "qwen3-tts-0.6b": {"size": "0.6b", "quant": "Q8_0", "codec": "Q8_0"},
    "qwen3-tts-0.6b-bf16": {"size": "0.6b", "quant": "BF16", "codec": "BF16"},
    "qwen3-tts-0.6b-q8": {"size": "0.6b", "quant": "Q8_0", "codec": "Q8_0"},
    "qwen3-tts-0.6b-q6": {"size": "0.6b", "quant": "Q6_K", "codec": "Q8_0"},
    "qwen3-tts-0.6b-q5": {"size": "0.6b", "quant": "Q5_K_M", "codec": "Q8_0"},
    "qwen3-tts-0.6b-q4": {"size": "0.6b", "quant": "Q4_K_M", "codec": "Q8_0"},
    "qwen3-tts-1.7b": {"size": "1.7b", "quant": "Q8_0", "codec": "Q8_0"},
    "qwen3-tts-1.7b-bf16": {"size": "1.7b", "quant": "BF16", "codec": "BF16"},
    "qwen3-tts-1.7b-q8": {"size": "1.7b", "quant": "Q8_0", "codec": "Q8_0"},
    "qwen3-tts-1.7b-q6": {"size": "1.7b", "quant": "Q6_K", "codec": "Q8_0"},
    "qwen3-tts-1.7b-q5": {"size": "1.7b", "quant": "Q5_K_M", "codec": "Q8_0"},
    "qwen3-tts-1.7b-q4": {"size": "1.7b", "quant": "Q4_K_M", "codec": "Q8_0"},
}
MODEL_ALIASES = {**FALLBACK_MODELS, **CATALOG_MODELS}
TALKER_PATTERN = CATALOG.get("talker_pattern", "qwen-talker-{size}-base-{quant}.gguf")
CODEC_PATTERN = CATALOG.get(
    "codec_pattern", "qwen-tokenizer-12hz-{quant}.gguf"
)
DEFAULT_CODEC = CATALOG.get("codec", "qwen-tokenizer-12hz-Q8_0.gguf")
DEFAULT_REPOSITORY = "Serveurperso/Qwen3-TTS-GGUF"
DOWNLOADABLE_QUANTS = frozenset({"BF16", "F32", "Q8_0", "Q4_K_M"})
QUANT_ALIASES = {
    "bf16": "BF16",
    "f32": "F32",
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
    "native_library": None,
    "backend": "auto",
    "use_fa": True,
    "clamp_fp16": False,
    "max_batch": 1,
    "codec": None,
    "binary": None,
    "codec_binary": None,
    "quantize_binary": None,
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
    "auto_download": True,
    "model_repository": DEFAULT_REPOSITORY,
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
        "timeout",
    }
)
# Aliases such as ``qwen3-tts-0.6b`` and ``qwen3-tts-0.6b-q8`` resolve to the
# same talker/codec files. The alias must not force a second resident model.
CACHE_KEY_EXCLUDED = PER_REQUEST_OPTIONS | {"model"}


@dataclass(frozen=True)
class QwenVoiceState:
    """Reference conditioning: speaker embedding plus optional ICL codes."""

    spk: Path
    rvq: Path | None = None
    ref_text: Path | None = None
    source: Path | None = None


def _download_model_file(path: Path, repository: str) -> None:
    """Download one missing GGUF into ``path.parent`` from Hugging Face."""
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise FileNotFoundError(
            f"Qwen3-TTS model file not found: {path}. Install huggingface_hub "
            "or pre-download the GGUF."
        ) from error
    logger.info("Downloading Qwen GGUF %s from %s", path.name, repository)
    try:
        hf_hub_download(
            repo_id=repository,
            filename=path.name,
            local_dir=str(path.parent),
        )
    except Exception as error:  # include upstream HF detail
        raise FileNotFoundError(
            f"Qwen3-TTS model file not found: {path}, and download from "
            f"{repository} failed: {error}"
        ) from error
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Qwen3-TTS download did not produce {path}")


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
        self._device = str(options["device"])
        self._backend = str(options.get("backend") or "auto")
        self._native = None
        native_library = options.get("native_library")
        if self._backend in ("auto", "native") and native_library:
            from providers.qwen3_native import QwenNativeBackend

            self._native = QwenNativeBackend(
                native_library,
                self._talker,
                self._codec,
                device=self._device,
                use_fa=bool(options.get("use_fa", True)),
                clamp_fp16=bool(options.get("clamp_fp16", False)),
                max_batch=int(options.get("max_batch", 1)),
                codec_chunk_sec=float(options.get("codec_chunk_dur") or 4.0),
            )
            self._binary = None
        elif self._backend == "native":
            raise FileNotFoundError(
                "Qwen native backend requested but native_library is not set"
            )
        else:
            self._binary = _resolve_executable(str(options["binary"]), "qwen-tts")
        self._codec_binary = _resolve_executable(
            str(options["codec_binary"]), "qwen-codec"
        )
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

    def close(self) -> None:
        if self._native is not None:
            self._native.close()

    def cancel(self) -> None:
        if self._native is not None:
            self._native.cancel()

    def synthesize(
        self,
        voice: QwenVoiceState,
        text: str,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[AudioChunk]:
        opts = {**self._options, **(options or {})}
        if self._native is not None:
            yield from self._native.synthesize(
                text,
                voice.spk,
                voice.rvq,
                voice.ref_text,
                options=opts,
            )
            return
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
        language = {
            "auto": "auto",
            "zh": "chinese",
            "en": "english",
            "ja": "japanese",
            "ko": "korean",
            "de": "german",
            "fr": "french",
            "ru": "russian",
            "pt": "portuguese",
            "es": "spanish",
            "it": "italian",
        }.get(language.lower(), language)
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
            return {
                str(key): str(value)
                for key, value in spec.items()
                if key in {"size", "quant", "codec"}
            }
        size = "1.7b" if "1.7b" in model else "0.6b"
        suffix = model.rsplit("-", 1)[-1].lower()
        return {"size": size, "quant": QUANT_ALIASES.get(suffix, "Q8_0")}

    def resolve_options(self, model: str, options: Mapping[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(options) - ALLOWED_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown qwen3-tts options: {', '.join(unknown)}")
        resolved = {**DEFAULT_OPTIONS, **options}
        backend = str(resolved.get("backend") or "auto").lower()
        if backend not in {"auto", "native", "cli"}:
            raise ValueError("qwen3-tts backend must be one of: auto, native, cli")
        resolved["backend"] = backend
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
            codec = spec.get("codec")
            if codec is None:
                codec_path = models / DEFAULT_CODEC
            elif codec.endswith(".gguf"):
                codec_path = models / codec
            else:
                codec_path = models / CODEC_PATTERN.format(quant=codec)
            resolved["codec"] = str(codec_path)
        if resolved["binary"] is None:
            resolved["binary"] = "build/qwen-tts"
        if resolved["codec_binary"] is None:
            resolved["codec_binary"] = "build/qwen-codec"
        return {"model": model, **resolved}

    def model_cache_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in options.items()
            if key not in CACHE_KEY_EXCLUDED
        }

    @staticmethod
    def _ensure_quantized(
        path: Path,
        model: str,
        options: Mapping[str, Any],
        repository: str,
    ) -> None:
        """Quantize the matching BF16 talker when a K-quant is not published."""
        spec = Qwen3TTSProvider._model_spec(model)
        quant = spec["quant"]
        if quant in DOWNLOADABLE_QUANTS:
            raise FileNotFoundError(
                f"Qwen3-TTS talker file not found: {path}. The repository "
                f"{repository} did not provide it."
            )
        base = path.with_name(
            TALKER_PATTERN.format(size=spec["size"], quant="BF16")
        )
        _download_model_file(base, repository)
        quantize = _resolve_executable(
            str(options.get("quantize_binary") or "build/quantize"), "quantize"
        )
        temporary = path.with_suffix(".gguf.tmp")
        logger.info("Quantizing %s -> %s (%s)", base.name, path.name, quant)
        try:
            result = subprocess.run(
                [str(quantize), str(base), str(temporary), quant],
                capture_output=True,
                timeout=1800,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Timed out quantizing Qwen talker {path}") from error
        if result.returncode != 0:
            temporary.unlink(missing_ok=True)
            detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
            raise RuntimeError(
                f"Could not quantize Qwen talker {path} to {quant}: "
                + ("\n".join(detail[-8:]) or "no stderr output")
            )
        temporary.replace(path)

    def _ensure_talker(
        self,
        path: Path,
        model: str,
        options: Mapping[str, Any],
        repository: str,
    ) -> None:
        if path.is_file() and path.stat().st_size > 0:
            return
        if not bool(options.get("auto_download", True)):
            raise FileNotFoundError(
                f"Qwen3-TTS talker file not found: {path}. Download the GGUF "
                "or set the provider option."
            )
        try:
            _download_model_file(path, repository)
        except FileNotFoundError:
            spec = self._model_spec(model)
            if spec["quant"] in DOWNLOADABLE_QUANTS:
                raise
            self._ensure_quantized(path, model, options, repository)

    def load_model(self, options: dict[str, Any]) -> Qwen3TTSModel:
        auto_download = bool(options.get("auto_download", True))
        repository = str(options.get("model_repository") or DEFAULT_REPOSITORY)
        model = str(options.get("model") or "")
        talker = Path(str(options["talker"]))
        self._ensure_talker(talker, model, options, repository)
        codec = Path(str(options["codec"]))
        if auto_download:
            _download_model_file(codec, repository)
        elif not codec.is_file():
            raise FileNotFoundError(
                f"Qwen3-TTS codec file not found: {codec}. Download the GGUF "
                "or set the provider option."
            )
        backend = str(options.get("backend") or "auto")
        native_library = options.get("native_library")
        if backend != "cli" and native_library:
            if not Path(str(native_library)).is_file():
                raise FileNotFoundError(
                    f"Qwen native library not found: {native_library}"
                )
        elif backend == "native":
            raise FileNotFoundError(
                "Qwen native backend requested but native_library is not set"
            )
        else:
            _resolve_executable(str(options["binary"]), "qwen-tts")
        _resolve_executable(str(options["codec_binary"]), "qwen-codec")
        return Qwen3TTSModel(options)
