"""FishAudio S1-mini provider: dual-AR language model + firefly codec.

The checkpoint is gated on Hugging Face. ``huggingface_hub`` picks up
``HF_TOKEN`` or the stored HF token; accept the model terms on the Hub once.
Weights are CC-BY-NC-SA-4.0 and fish-speech code is under the Fish Audio
Research License: research/non-commercial use only.
"""

import logging
import platform
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from torch.nn.attention import SDPBackend, sdpa_kernel

from tts_core.audio import AudioChunk
from tts_core.provider import SynthesisModel, TTSProvider, VoiceState

from ._upstream.inference import decode_one_token_ar, generate_segments
from ._upstream.llama import DualARTransformer
from ._upstream.modded_dac import DAC, WindowLimitedTransformer
from ._upstream.modded_dac import ModelArgs as TransformerArgs
from ._upstream.rvq import DownsampleResidualVectorQuantize

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "fishaudio/s1-mini"
CHECKPOINT_ALIASES = {"s1-mini", "fish-s1-mini", "fishaudio/s1-mini"}
SAMPLE_RATE = 44_100

DEFAULT_OPTIONS: dict[str, Any] = {
    "checkpoint": None,
    "device": "auto",
    "dtype": "bfloat16",
    "quantize": "none",
    "reference_text": "",
    "max_reference_seconds": 30.0,
    "temperature": 0.8,
    "top_p": 0.8,
    "repetition_penalty": 1.1,
    "max_new_tokens": 1024,
    "chunk_length": 300,
    "iterative_prompt": True,
    "seed": None,
    "compile": False,
    "threads": None,
}
ALLOWED_OPTIONS = frozenset(DEFAULT_OPTIONS)


@dataclass(frozen=True)
class FishVoice:
    """Reference conditioning: codec tokens plus the reference transcript."""

    codes: torch.Tensor | None
    text: str


class FishS1MiniModel(SynthesisModel):
    def __init__(
        self,
        model: DualARTransformer,
        codec: DAC,
        *,
        device: torch.device,
        options: Mapping[str, Any],
    ):
        self._model = model
        self._codec = codec
        self._device = device
        self._options = dict(options)
        self._decode_one_token = decode_one_token_ar
        if options["compile"]:
            compile_kwargs: dict[str, Any] = {"fullgraph": True, "dynamic": True}
            if device.type == "cuda":
                compile_kwargs["mode"] = "reduce-overhead"
            self._decode_one_token = torch.compile(
                decode_one_token_ar, **compile_kwargs
            )
            logger.info("torch.compile enabled for the Fish decode step")

    @property
    def sample_rate(self) -> int:
        return self._codec.sample_rate

    def load_voice(self, source: str | Path | None) -> VoiceState:
        text = str(self._options["reference_text"])
        if source is None:
            return FishVoice(None, text)

        path = Path(source)
        label_path = path.with_suffix(".lab")
        if label_path.is_file():
            text = label_path.read_text(encoding="utf-8").strip()

        audio = _load_audio(path, self.sample_rate)
        limit = int(float(self._options["max_reference_seconds"]) * self.sample_rate)
        audio = audio[:limit]
        if audio.size == 0:
            raise ValueError(f"Reference audio {path} is empty")

        with torch.inference_mode(), _attention_context(self._device):
            samples = torch.from_numpy(audio).to(self._device)[None, None, :]
            lengths = torch.tensor(
                [samples.shape[2]], device=self._device, dtype=torch.long
            )
            codes, _ = self._codec.encode(samples, lengths)

        return FishVoice(codes[0].cpu(), text)

    def synthesize(
        self,
        voice: FishVoice,
        text: str,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[AudioChunk]:
        generation = {**self._options, **(options or {})}
        with torch.inference_mode(), _attention_context(self._device):
            segments = generate_segments(
                model=self._model,
                text=text,
                prompt_text=voice.text if voice.codes is not None else None,
                prompt_tokens=voice.codes,
                max_new_tokens=int(generation["max_new_tokens"]),
                top_p=float(generation["top_p"]),
                repetition_penalty=float(generation["repetition_penalty"]),
                temperature=float(generation["temperature"]),
                chunk_length=int(generation["chunk_length"]),
                iterative_prompt=bool(generation["iterative_prompt"]),
                seed=generation["seed"],
                decode_one_token=self._decode_one_token,
            )
            for codes in segments:
                yield self._decode(codes)

    @torch.inference_mode()
    def _decode(self, codes: torch.Tensor) -> AudioChunk:
        codes = codes.to(self._device).long()
        lengths = torch.tensor([codes.shape[1]], device=self._device, dtype=torch.long)
        with _attention_context(self._device):
            audio, _ = self._codec.decode(codes[None], lengths)
        return audio[0, 0].float().cpu().numpy()

    def export_voice(self, voice: FishVoice, destination: Path) -> None:
        if voice.codes is None:
            raise ValueError("An unconditioned voice has nothing to export")
        audio = self._decode(voice.codes)
        sf.write(destination, audio, self.sample_rate)
        label = destination.with_suffix(".lab")
        if voice.text:
            label.write_text(voice.text, encoding="utf-8")


class FishS1MiniProvider(TTSProvider):
    """FishAudio S1-mini: 13-language TTS with reference-audio voice cloning."""

    name: ClassVar[str] = "fish-s1-mini"
    available_models: ClassVar[tuple[str, ...]] = ("s1-mini",)
    voice_extension: ClassVar[str] = "wav"
    default_voice: ClassVar[str | None] = "default"
    default_voice_source: ClassVar[str | None] = None

    def resolve_options(self, model: str, options: Mapping[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(options) - ALLOWED_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown fish-s1-mini options: {', '.join(unknown)}")

        resolved = {**DEFAULT_OPTIONS, **options}
        resolved["checkpoint"] = resolved["checkpoint"] or _checkpoint_uri(model)
        resolved["device"] = _resolve_device(str(resolved["device"]))
        resolved["dtype"] = _resolve_dtype(str(resolved["dtype"]))
        if resolved["quantize"] not in ("none", "int8", "int8-weight"):
            raise ValueError("quantize must be 'none', 'int8', or 'int8-weight'")
        if not 0 < float(resolved["top_p"]) <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not 0 < float(resolved["repetition_penalty"]) < 2:
            raise ValueError("repetition_penalty must be in (0, 2)")
        if not 0 < float(resolved["temperature"]) < 2:
            raise ValueError("temperature must be in (0, 2)")
        if int(resolved["chunk_length"]) <= 0:
            raise ValueError("chunk_length must be positive")
        if int(resolved["max_new_tokens"]) < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if float(resolved["max_reference_seconds"]) <= 0:
            raise ValueError("max_reference_seconds must be positive")
        if resolved["seed"] is not None and not isinstance(resolved["seed"], int):
            raise ValueError("seed must be an integer or null")
        if not isinstance(resolved["compile"], bool):
            raise ValueError("compile must be a boolean")  # noqa: TRY004
        if resolved["threads"] is not None and int(resolved["threads"]) < 1:
            raise ValueError("threads must be a positive integer or null")
        return resolved

    def model_cache_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        # Sampling options vary per request; checkpoint and voice conditioning
        # identify the loaded model (and its voice cache).
        return {
            key: options[key]
            for key in (
                "checkpoint",
                "device",
                "dtype",
                "quantize",
                "compile",
                "threads",
                "reference_text",
                "max_reference_seconds",
            )
        }

    def load_model(self, options: Mapping[str, Any]) -> SynthesisModel:
        checkpoint = str(options["checkpoint"])
        path = Path(checkpoint).expanduser()
        if not (path / "model.pth").is_file():
            if checkpoint.startswith(("/", "~", ".")):
                raise FileNotFoundError(f"No model.pth under {path}")
            logger.info("Downloading %s", checkpoint)
            path = Path(snapshot_download(checkpoint))

        device = torch.device(str(options["device"]))
        dtype = getattr(torch, str(options["dtype"]))
        if options["threads"] is not None:
            torch.set_num_threads(int(options["threads"]))
            logger.info("Set torch CPU threads to %s", options["threads"])
        logger.info("Loading FishAudio S1-mini from %s on %s", path, device)

        model = DualARTransformer.from_pretrained(path)
        model = model.to(device=device, dtype=dtype)

        if options["quantize"] != "none":
            weight_only = options["quantize"] == "int8-weight"
            if not weight_only and dtype is not torch.float32:
                model = model.to(dtype=torch.float32)
            model = _apply_int8(model, weight_only=weight_only)
            logger.info(
                "Applied torchao int8 %s quantization to the AR model",
                "weight-only" if weight_only else "dynamic",
            )

        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        model.eval()

        codec = _build_codec()
        state = torch.load(
            path / "codec.pth", map_location="cpu", mmap=True, weights_only=True
        )
        if "state_dict" in state:
            state = state["state_dict"]
        if any(key.startswith("generator.") for key in state):
            state = {
                key.removeprefix("generator."): value
                for key, value in state.items()
                if key.startswith("generator.")
            }
        missing, unexpected = codec.load_state_dict(state, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(
                "codec.pth does not match the firefly codec: "
                f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}"
            )
        codec.eval().to(device)
        logger.info("FishAudio S1-mini ready (quantize=%s)", options["quantize"])

        return FishS1MiniModel(model, codec, device=device, options=options)


def _checkpoint_uri(model: str) -> str:
    if model in ("", "default") or model in CHECKPOINT_ALIASES:
        return DEFAULT_CHECKPOINT
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        return str(candidate)
    # A bare word (the engine's default "english", for example) is not a repo id.
    return model if "/" in model else DEFAULT_CHECKPOINT


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _attention_context(device: torch.device):
    """Force the math SDPA kernel on XPU.

    Intel's SDPA kernel fails to create a primitive for masked attention on
    Arc GPUs; the math kernel is supported and the codec/AR attention is a
    small fraction of total runtime.
    """
    if device.type == "xpu":
        return sdpa_kernel(SDPBackend.MATH)
    return nullcontext()


def _resolve_device(name: str) -> str:
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _xpu_available():
            return "xpu"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if name.startswith("xpu") and not _xpu_available():
        raise ValueError("XPU is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")
    return name


def _resolve_dtype(name: str) -> str:
    mapping = {
        "bfloat16": "bfloat16",
        "float16": "float16",
        "half": "float16",
        "float32": "float32",
        "fp32": "float32",
    }
    try:
        return mapping[name]
    except KeyError:
        raise ValueError("dtype must be one of bfloat16, float16, float32") from None


def _apply_int8(
    model: torch.nn.Module, *, weight_only: bool = False
) -> torch.nn.Module:
    try:
        from torchao.quantization import (
            Int8DynamicActivationInt8WeightConfig,
            Int8WeightOnlyConfig,
            quantize_,
        )
    except ImportError:
        if weight_only:
            raise ValueError("quantize='int8-weight' requires torchao") from None
        logger.warning("torchao not installed; falling back to torch.ao int8")
        torch.backends.quantized.engine = (
            "qnnpack" if platform.machine() in ("arm64", "aarch64") else "fbgemm"
        )
        return torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    config = (
        Int8WeightOnlyConfig()
        if weight_only
        else Int8DynamicActivationInt8WeightConfig()
    )
    quantize_(model, config)
    return model


def _build_codec() -> DAC:
    """Instantiate the 44.1 kHz firefly codec from modded_dac_vq.yaml."""
    general = partial(
        TransformerArgs,
        block_size=16384,
        n_local_heads=-1,
        head_dim=64,
        rope_base=10000,
        norm_eps=1e-5,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        channels_first=True,
    )
    quantizer_args = TransformerArgs(
        block_size=4096,
        n_layer=8,
        n_head=16,
        dim=1024,
        intermediate_size=3072,
        n_local_heads=-1,
        head_dim=64,
        rope_base=10000,
        norm_eps=1e-5,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        channels_first=True,
    )

    def windowed_transformer() -> WindowLimitedTransformer:
        return WindowLimitedTransformer(
            causal=True, window_size=128, input_dim=1024, config=quantizer_args
        )

    quantizer = DownsampleResidualVectorQuantize(
        input_dim=1024,
        n_codebooks=9,
        codebook_size=1024,
        codebook_dim=8,
        quantizer_dropout=0.5,
        downsample_factor=(2, 2),
        pre_module=windowed_transformer(),
        post_module=windowed_transformer(),
    )
    return DAC(
        encoder_dim=64,
        encoder_rates=[2, 4, 8, 8],
        decoder_dim=1536,
        decoder_rates=[8, 8, 4, 2],
        quantizer=quantizer,
        sample_rate=SAMPLE_RATE,
        encoder_transformer_layers=[0, 0, 0, 4],
        decoder_transformer_layers=[4, 0, 0, 0],
        transformer_general_config=general,
    )


def _load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Decode any ffmpeg-supported file to mono float32 at ``sample_rate``."""
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
    return np.frombuffer(result.stdout, dtype="<f4").copy()
