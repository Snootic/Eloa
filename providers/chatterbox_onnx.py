"""Chatterbox-Turbo via its official ONNX export on OpenVINO.

No torch dependency: the four ONNX graphs run through OpenVINO, the AR loop
and sampling stay in NumPy. The language model and token embedding run on the
OpenVINO GPU; the speech encoder and decoder stay on CPU because their fp16
graphs either fail to compile or crash on the Intel GPU driver used here.
"""

import logging
import os
import shutil
import subprocess
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import openvino as ov

# Create the Core before importing transformers/torch; otherwise OpenVINO
# loses the Intel GPU to torch's device initialization.
CORE = ov.Core()

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from tts_core.audio import AudioChunk
from tts_core.provider import SynthesisModel, TTSProvider, VoiceState

logger = logging.getLogger(__name__)

REPO = "ResembleAI/chatterbox-turbo-ONNX"
SAMPLE_RATE = 24_000
START_TOKEN = 6561
STOP_TOKEN = 6562
SILENCE_TOKEN = 4299
NUM_KV_HEADS = 16
HEAD_DIM = 64
GRAPH_FILES = (
    "embed_tokens",
    "language_model",
    "speech_encoder",
    "conditional_decoder",
)

DEFAULT_OPTIONS: dict[str, Any] = {
    "checkpoint": None,
    "device": "AUTO",
    "temperature": 0.8,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "max_new_tokens": 1024,
    "seed": None,
}
ALLOWED_OPTIONS = frozenset(DEFAULT_OPTIONS)


@dataclass(frozen=True)
class ChatterboxOVVoice:
    path: Path
    features: np.ndarray
    tokens: np.ndarray
    speaker_embeddings: np.ndarray
    speaker_features: np.ndarray


class ChatterboxOVModel(SynthesisModel):
    def __init__(self, directory: Path, device: str, options: Mapping[str, Any]):
        self._dir = directory
        self._options = dict(options)
        self._core = CORE
        self._device = self._pick_device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(str(directory))
        self._embed = self._compile("embed_tokens", (self._device, "CPU"))
        self._lm = self._compile("language_model", (self._device, "CPU"))
        # These two fp16 graphs are not usable on this Intel GPU driver.
        self._encoder = self._compile("speech_encoder", ("CPU",))
        self._decoder = self._compile("conditional_decoder", ("CPU",))
        self._lm_inputs = [i.get_any_name() for i in self._lm.inputs]
        self._lm_outputs = [o.get_any_name() for o in self._lm.outputs]
        self._logits_name = next(name for name in self._lm_outputs if "logits" in name)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def _pick_device(self, device: str) -> str:
        devices = list(self._core.available_devices)
        requested = device.upper()
        if requested == "AUTO":
            return "GPU" if "GPU" in devices else "CPU"
        if requested not in devices:
            raise ValueError(
                f"OpenVINO device {device!r} is not available; have {devices}"
            )
        return requested

    def _compile(self, name: str, devices: tuple[str, ...]):
        model = self._core.read_model(str(self._dir / "onnx" / f"{name}_fp16.onnx"))
        for device in devices:
            try:
                compiled = self._core.compile_model(model, device)
                logger.info("Chatterbox %s: %s", name, device)
                return compiled
            except Exception:
                logger.debug("Chatterbox %s failed on %s", name, device, exc_info=True)
        raise RuntimeError(f"Could not compile {name} on {devices}")

    def load_voice(self, source: str | Path | None) -> VoiceState:
        if source is None:
            raise ValueError(
                "Chatterbox Turbo requires a reference clip for voice cloning"
            )
        audio = _load_audio(Path(source), SAMPLE_RATE)
        result = {
            key.get_any_name(): value
            for key, value in self._encoder({"audio_values": audio}).items()
        }
        return ChatterboxOVVoice(
            path=Path(source),
            features=np.asarray(result["audio_features"], dtype=np.float16),
            tokens=np.asarray(result["audio_tokens"], dtype=np.int64),
            speaker_embeddings=np.asarray(
                result["speaker_embeddings"], dtype=np.float32
            ),
            speaker_features=np.asarray(result["speaker_features"], dtype=np.float32),
        )

    def synthesize(
        self,
        voice: ChatterboxOVVoice,
        text: str,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[AudioChunk]:
        opts = {**self._options, **(options or {})}
        ids = self._tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)
        embedding = next(iter(self._embed({"input_ids": ids}).values()))
        inputs_embeds = np.concatenate(
            [voice.features, np.asarray(embedding, dtype=np.float16)], axis=1
        )

        batch, seq, _ = inputs_embeds.shape
        caches = {
            name: np.zeros((batch, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float16)
            for name in self._lm_inputs
            if "past_key_values" in name
        }
        attention_mask = np.ones((batch, seq), dtype=np.int64)
        position_ids = np.arange(seq, dtype=np.int64).reshape(1, -1)
        generated = np.array([[START_TOKEN]], dtype=np.int64)
        rng = np.random.default_rng(opts["seed"])
        temperature = float(opts["temperature"])
        top_p = float(opts["top_p"])
        penalty = float(opts["repetition_penalty"])
        limit = int(opts["max_new_tokens"]) or 1024

        for _ in range(limit):
            feed = {
                **caches,
                "inputs_embeds": inputs_embeds.astype(np.float16),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
            result = dict(
                zip(
                    self._lm_outputs,
                    list(self._lm([feed[name] for name in self._lm_inputs]).values()),
                )
            )
            logits = np.asarray(result[self._logits_name], dtype=np.float32)[
                :, -1, :
            ].copy()

            scores = np.take_along_axis(logits, generated, axis=1)
            scores = np.where(scores < 0, scores * penalty, scores / penalty)
            np.put_along_axis(logits, generated, scores, axis=1)

            probs = logits[0] / max(temperature, 1e-5)
            probs = np.exp(probs - probs.max())
            probs /= probs.sum()
            order = np.argsort(-probs)
            cutoff = np.searchsorted(np.cumsum(probs[order]), top_p) + 1
            keep = order[:cutoff]
            filtered = probs[keep]
            filtered /= filtered.sum()
            token = np.array([[rng.choice(keep, p=filtered)]], dtype=np.int64)

            generated = np.concatenate([generated, token], axis=1)
            if int(token[0, 0]) == STOP_TOKEN:
                break

            attention_mask = np.concatenate(
                [attention_mask, np.ones((batch, 1), dtype=np.int64)], axis=1
            )
            position_ids = position_ids[:, -1:] + 1
            caches = {
                name: np.asarray(
                    result[name.replace("past_key_values", "present")],
                    dtype=np.float16,
                )
                for name in caches
            }
            inputs_embeds = np.asarray(
                next(iter(self._embed({"input_ids": token}).values())),
                dtype=np.float16,
            )

        speech = np.concatenate(
            [
                voice.tokens,
                generated[:, 1:-1],
                np.full((1, 3), SILENCE_TOKEN, dtype=np.int64),
            ],
            axis=1,
        )
        waveform = next(
            iter(
                self._decoder(
                    {
                        "speech_tokens": speech,
                        "speaker_embeddings": voice.speaker_embeddings,
                        "speaker_features": voice.speaker_features,
                    }
                ).values()
            )
        )
        yield np.asarray(waveform, dtype=np.float32).reshape(-1)

    def export_voice(self, voice: ChatterboxOVVoice, destination: Path) -> None:
        shutil.copyfile(voice.path, destination)


class ChatterboxONNXProvider(TTSProvider):
    """Chatterbox-Turbo ONNX on OpenVINO (torch-free)."""

    name: ClassVar[str] = "chatterbox-turbo-ov"
    available_models: ClassVar[tuple[str, ...]] = ("chatterbox-turbo",)
    voice_extension: ClassVar[str] = "wav"
    default_voice: ClassVar[str | None] = None
    default_voice_source: ClassVar[str | None] = None

    def resolve_options(self, model: str, options: Mapping[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(options) - ALLOWED_OPTIONS)
        if unknown:
            raise ValueError(
                f"Unknown chatterbox-turbo-ov options: {', '.join(unknown)}"
            )
        resolved = {**DEFAULT_OPTIONS, **options}
        resolved["checkpoint"] = _ensure_checkpoint(resolved["checkpoint"] or REPO)
        resolved["device"] = str(resolved["device"]).upper()
        if resolved["device"] not in ("AUTO", "GPU", "CPU"):
            raise ValueError("device must be AUTO, GPU, or CPU")
        if not 0 < float(resolved["top_p"]) <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not 0 < float(resolved["temperature"]) < 2:
            raise ValueError("temperature must be in (0, 2)")
        if float(resolved["repetition_penalty"]) <= 0:
            raise ValueError("repetition_penalty must be positive")
        return resolved

    def model_cache_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        return {key: options[key] for key in ("checkpoint", "device")}

    def load_model(self, options: Mapping[str, Any]) -> SynthesisModel:
        logger.info("Loading Chatterbox-Turbo ONNX from %s", options["checkpoint"])
        return ChatterboxOVModel(
            Path(str(options["checkpoint"])),
            str(options["device"]),
            options,
        )


def _cache_dir() -> Path:
    base = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "tts-core" / "chatterbox-turbo-onnx"


def _ensure_checkpoint(checkpoint: str) -> str:
    """Return a real (non-symlink) ONNX directory with external data files."""
    path = Path(checkpoint).expanduser()
    data = path / "onnx" / "language_model_fp16.onnx_data"
    if data.is_file() and not data.is_symlink():
        return str(path)

    target = _cache_dir()
    if (target / "onnx" / "language_model_fp16.onnx_data").is_file():
        return str(target)

    logger.info("Downloading Chatterbox-Turbo ONNX graphs")
    patterns = [
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
        *[f"onnx/{name}_fp16.onnx" for name in GRAPH_FILES],
        *[f"onnx/{name}_fp16.onnx_data" for name in GRAPH_FILES],
    ]
    snapshot = Path(snapshot_download(REPO, allow_patterns=patterns))
    target.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        source = snapshot / pattern
        destination = target / pattern
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Hugging Face snapshots are symlinks into the blob store; OpenVINO
        # rejects external data paths that escape the model directory.
        shutil.copyfile(os.path.realpath(source), destination)
    return str(target)


def _load_audio(path: Path, sample_rate: int) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Could not decode reference audio {path}: {error}")
    return np.frombuffer(result.stdout, dtype="<f4")[None, :].astype(np.float16)
