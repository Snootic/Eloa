"""Chatterbox Multilingual TTS through codec.cpp GGUF (Vulkan/CUDA/CPU).

The model is loaded once through codec.cpp's Python bindings and kept
resident. The backbone (T3) runs in the bundled llama.cpp; the codec GGUF
runs the S3T/S3Gen stack. Both files come from
``BricksDisplay/Chatterbox-Multilingual-TTS-GGUF``.

Text uses Chatterbox's MTL convention: lowercase + NFKD and a ``[xx]``
language prefix. Reference voices are turned into a reusable speaker with
codec.cpp's CAMPPlus/VoiceEncoder path. Chatterbox can generate one segment
at a time only after it is complete, so ``segment_bytes`` controls the
latency/quality split.
"""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import soundfile as sf

from tts_core.audio import AudioChunk, as_mono_float32
from tts_core.provider import SynthesisModel, TTSProvider

SAMPLE_RATE = 24_000
MODEL_ALIASES: dict[str, str] = {
    "chatterbox-mtl": "q8_0",
    "chatterbox-mtl-q8": "q8_0",
    "chatterbox-mtl-q8_0": "q8_0",
    "chatterbox-mtl-q4": "q4_k_m",
    "chatterbox-mtl-q4_k_m": "q4_k_m",
    "chatterbox-mtl-f16": "f16",
}
QUANT_ALIASES = {
    "q8": "q8_0",
    "q8_0": "q8_0",
    "q4": "q4_k_m",
    "q4_k_m": "q4_k_m",
    "f16": "f16",
    "full": "f16",
}

DEFAULT_OPTIONS: dict[str, Any] = {
    "models_dir": None,
    "quant": None,
    "backbone": None,
    "codec": None,
    "use_gpu": True,
    "n_threads": 0,
    "max_frames": 0,
    "language": "en",
    "segment_bytes": 100,
    "emotion": 0.5,
    "flow_steps": None,
    "min_frames": None,
    "lm_guidance": None,
    "flow_guidance": None,
    "seed": 7,
    "temp": None,
    "top_k": None,
    "top_p": None,
    "min_p": None,
    "penalty_last_n": None,
    "penalty_repeat": None,
    "python_path": None,
    "library": None,
    "backbone_library": None,
    "kv_overrides": None,
}
ALLOWED_OPTIONS = frozenset(DEFAULT_OPTIONS)
PER_REQUEST_OPTIONS = frozenset(
    {
        "language",
        "segment_bytes",
        "flow_steps",
        "min_frames",
        "lm_guidance",
        "flow_guidance",
        "seed",
        "temp",
        "top_k",
        "top_p",
        "min_p",
        "penalty_last_n",
        "penalty_repeat",
    }
)
SAMPLING_OPTIONS = (
    "seed",
    "temp",
    "top_k",
    "top_p",
    "min_p",
    "penalty_last_n",
    "penalty_repeat",
)
SYNTH_OPTIONS = (
    "segment_bytes",
    "flow_steps",
    "min_frames",
    "lm_guidance",
    "flow_guidance",
)


def _load_codec_cpp(options: Mapping[str, Any]):
    """Import codec_cpp, optionally from a source checkout and shared libs."""
    python_path = options.get("python_path")
    if python_path:
        path = str(Path(str(python_path)).expanduser().resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
    for key, env_name in (
        ("library", "CODEC_CPP_LIBRARY"),
        ("backbone_library", "CODEC_CPP_BACKBONE_LIBRARY"),
    ):
        value = options.get(key)
        if value:
            os.environ[env_name] = str(Path(str(value)).expanduser())
    try:
        return importlib.import_module("codec_cpp")
    except ImportError as error:
        raise ImportError(
            "chatterbox-gguf needs codec_cpp; build codec.cpp and set "
            "python_path/library/backbone_library, or install "
            "`pip install git+https://github.com/mybigday/codec.cpp`"
        ) from error


def _transcript_for(source: Path | None, name: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if source is not None:
        candidates.extend((source.with_suffix(".txt"), source.with_suffix(".lab")))
        if source.suffix and source.stem.endswith(".ref"):
            candidates.append(source.with_name(f"{source.stem[:-4]}.txt"))
            candidates.append(source.with_name(f"{source.stem[:-4]}.lab"))
    if name:
        candidates.extend(
            (Path("voices") / f"{name}.txt", Path("voices") / f"{name}.lab")
        )
    for candidate in candidates:
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                return candidate
    return None


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, sf.LibsndfileError):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            converted = Path(tmp.name)
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(path),
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    str(converted),
                ],
                capture_output=True,
                timeout=120,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Could not decode reference audio {path}: "
                    f"{result.stderr.decode('utf-8', 'replace').strip()}"
                )
            audio, rate = sf.read(converted, dtype="float32", always_2d=True)
        finally:
            converted.unlink(missing_ok=True)
    return np.ascontiguousarray(audio.mean(axis=1), dtype=np.float32), int(rate)


def _prepare_text(text: str, language: str) -> str:
    language = str(language or "en").strip().lower() or "en"
    match = re.match(r"^\s*\[([A-Za-z]{2,3})\]\s*", text)
    if match:
        language = match.group(1).lower()
        text = text[match.end() :]
    return f"[{language}]" + unicodedata.normalize("NFKD", text.lower())


@dataclass(frozen=True)
class ChatterboxVoiceState:
    name: str
    speaker: Any | None
    source: Path | None = None


class ChatterboxGGUFModel(SynthesisModel):
    def __init__(self, tts, options: Mapping[str, Any]):
        self._tts = tts
        self._options = dict(options)

    @property
    def sample_rate(self) -> int:
        return int(self._tts.audio_lm.caps.sample_rate)

    def load_voice(
        self, source: str | Path | None, *, name: str | None = None
    ) -> ChatterboxVoiceState:
        voice_name = name or "default"
        if source is None:
            return ChatterboxVoiceState(voice_name, None)
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Reference audio not found: {path}")
        if path.suffix.lower() == ".gguf":
            return ChatterboxVoiceState(
                voice_name,
                self._tts.audio_lm.load_voice(path),
                source=path,
            )
        audio, sample_rate = _read_audio(path)
        transcript_path = _transcript_for(path, voice_name)
        transcript = (
            transcript_path.read_text(encoding="utf-8", errors="replace").strip()
            if transcript_path
            else None
        )
        speaker = self._tts.audio_lm.create_voice(audio, sample_rate, transcript)
        return ChatterboxVoiceState(voice_name, speaker, source=path)

    def _synth_options(self, options: Mapping[str, Any] | None) -> dict[str, Any]:
        opts = {**self._options, **(options or {})}
        result: dict[str, Any] = {}
        for key in SYNTH_OPTIONS:
            if opts.get(key) is not None:
                result[key] = opts[key]
        for key in SAMPLING_OPTIONS:
            if opts.get(key) is not None:
                result[key] = opts[key]
        return result

    def synthesize(
        self,
        voice: ChatterboxVoiceState,
        text: str,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[AudioChunk]:
        opts = {**self._options, **(options or {})}
        prepared = _prepare_text(text, str(opts.get("language") or "en"))
        synth_options = self._synth_options(options)
        voice_arg: Any
        if voice.speaker is not None:
            voice_arg = voice.speaker
        else:
            voice_arg = voice.name if voice.name != "default" else None
        # Chatterbox emits a finished segment at a time; codec_cpp.stream
        # yields those pieces as they become final.
        for audio, _rate in self._tts.stream(
            prepared, voice=voice_arg, **synth_options
        ):
            chunk = as_mono_float32(audio)
            if chunk.size:
                yield chunk

    def synthesize_all(
        self,
        voice: ChatterboxVoiceState,
        text: str,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> AudioChunk:
        opts = {**self._options, **(options or {})}
        prepared = _prepare_text(text, str(opts.get("language") or "en"))
        synth_options = self._synth_options(options)
        synth_options["segment_bytes"] = int(opts.get("segment_bytes") or 0)
        voice_arg: Any = (
            voice.speaker
            if voice.speaker is not None
            else (voice.name if voice.name != "default" else None)
        )
        audio, _rate = self._tts.synthesize(prepared, voice=voice_arg, **synth_options)
        return as_mono_float32(audio)

    def export_voice(self, voice: ChatterboxVoiceState, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        speaker = voice.speaker
        if speaker is None:
            speaker = self._tts.audio_lm.voice(voice.name)
        # codec.cpp speaker names are restricted to [a-z0-9_].
        speaker_name = (
            re.sub(r"[^a-z0-9_]", "_", (voice.name or destination.stem).lower())[:32]
            or "voice"
        )
        self._tts.audio_lm.save_voice(speaker, destination, speaker_name)

    def close(self) -> None:
        self._tts.close()


class ChatterboxGGUFProvider(TTSProvider):
    """Chatterbox Multilingual through codec.cpp, resident and GPU-capable."""

    name: ClassVar[str] = "chatterbox-gguf"
    voice_extension: ClassVar[str] = "gguf"
    available_models: ClassVar[tuple[str, ...]] = tuple(MODEL_ALIASES)
    default_voice: ClassVar[str | None] = "default"
    default_voice_source: ClassVar[str | None] = None
    builtin_voices: ClassVar[tuple[str, ...]] = ("default",)

    @staticmethod
    def _quant(model: str, options: Mapping[str, Any]) -> str:
        alias_quant = MODEL_ALIASES.get(model)
        explicit = options.get("quant")
        # A named quantization alias wins over the deployment default; the
        # explicit quant is for the generic alias or unknown model names.
        if alias_quant is not None and model != "chatterbox-mtl":
            quant = alias_quant
        elif explicit:
            quant = QUANT_ALIASES.get(str(explicit).lower(), str(explicit).lower())
        elif alias_quant is not None:
            quant = alias_quant
        else:
            suffix = model.rsplit("-", 1)[-1].lower()
            quant = QUANT_ALIASES.get(suffix, model)
        if quant not in {"q8_0", "q4_k_m", "f16"}:
            raise ValueError(f"Unknown Chatterbox quantization: {quant}")
        return quant

    def resolve_options(self, model: str, options: Mapping[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(options) - ALLOWED_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown chatterbox-gguf options: {', '.join(unknown)}")
        resolved = {**DEFAULT_OPTIONS, **options}
        quant = self._quant(model, resolved)
        models = Path(str(resolved["models_dir"] or "models/chatterbox")).expanduser()
        backbone = (
            Path(str(resolved["backbone"])).expanduser()
            if resolved["backbone"]
            else models / f"chatterbox-mtl-{quant}.gguf"
        )
        codec = (
            Path(str(resolved["codec"])).expanduser()
            if resolved["codec"]
            else models / f"codec-{quant}.gguf"
        )
        resolved.update(
            model=model,
            quant=quant,
            models_dir=str(models),
            backbone=str(backbone),
            codec=str(codec),
            language=str(resolved["language"] or "en").lower(),
        )
        return resolved

    def model_cache_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            key: value
            for key, value in options.items()
            if key not in PER_REQUEST_OPTIONS
        }

    def load_model(self, options: dict[str, Any]) -> ChatterboxGGUFModel:
        for key in ("backbone", "codec"):
            path = Path(str(options[key]))
            if not path.is_file():
                raise FileNotFoundError(
                    f"Chatterbox {key} GGUF not found: {path}. Download "
                    "chatterbox-mtl-*.gguf and codec-*.gguf from "
                    "BricksDisplay/Chatterbox-Multilingual-TTS-GGUF."
                )
        codec_cpp = _load_codec_cpp(options)
        overrides = dict(options.get("kv_overrides") or {})
        if options.get("emotion") is not None:
            overrides.setdefault(
                "codec.lm.chatterbox.builtin.emotion_adv",
                float(options["emotion"]),
            )
        tts = codec_cpp.TTS(
            str(options["backbone"]),
            str(options["codec"]),
            use_gpu=bool(options.get("use_gpu", True)),
            n_threads=int(options.get("n_threads") or 0),
            max_frames=int(options.get("max_frames") or 0),
            kv_overrides=overrides or None,
        )
        return ChatterboxGGUFModel(tts, options)
