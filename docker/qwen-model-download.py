#!/usr/bin/env python3
"""Seed voices and optionally pre-download Qwen GGUF files, then exec the app.

The CUDA/Vulkan images run this as their entrypoint. Model files are downloaded
on the first request by the provider by default; set ``QWEN_PRELOAD=1`` to
restore startup downloads. Existing files are left alone; the default
repository is Serveurperso/Qwen3-TTS-GGUF. Override with QWEN_GGUF_REPO,
QWEN_GGUF_FILES (comma-separated) or QWEN_GGUF_QUANT.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

DEFAULT_REPO = "Serveurperso/Qwen3-TTS-GGUF"
QUANT_NAMES = {
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


def _truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _quant_from_model(model: str) -> str:
    match = re.search(r"-(q[0-9](?:_[a-z0-9]+)*|bf16|f32)$", model.lower())
    if match:
        return QUANT_NAMES.get(match.group(1), "BF16")
    explicit = os.getenv("QWEN_GGUF_QUANT", "").lower()
    return QUANT_NAMES.get(explicit, "BF16")


def _files_for_model(model: str) -> list[str]:
    extra = [
        name.strip()
        for name in os.getenv("QWEN_GGUF_FILES", "").split(",")
        if name.strip()
    ]
    if extra:
        return extra
    size = "1.7b" if "1.7b" in model.lower() else "0.6b"
    requested = _quant_from_model(model)
    quant = requested
    if quant not in {"BF16", "F32", "Q8_0", "Q4_K_M"}:
        # Q6/Q5 are generated locally from BF16 by the provider.
        quant = "BF16"
    codec_quant = "BF16" if requested in {"BF16", "F32"} else "Q8_0"
    return [
        f"qwen-talker-{size}-base-{quant}.gguf",
        os.getenv(
            "QWEN_TOKENIZER_GGUF", f"qwen-tokenizer-12hz-{codec_quant}.gguf"
        ),
    ]


def _seed_base_voices() -> None:
    source = Path(os.getenv("TTS_BASE_VOICES_DIR", "/app/base-voices"))
    if not source.is_dir():
        return
    target = Path(os.getenv("TTS_VOICES_DIR") or "/app/voices")
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        destination = target / item.name
        if item.is_file() and not destination.exists():
            shutil.copy2(item, destination)
            print(f"[voices] installed {item.name}", flush=True)


def main() -> None:
    _seed_base_voices()
    provider = os.getenv("TTS_PROVIDER", "qwen3-tts")
    if provider == "qwen3-tts" and _truthy("QWEN_PRELOAD"):
        models_dir = Path(os.getenv("TTS_MODELS_DIR", "/models"))
        models_dir.mkdir(parents=True, exist_ok=True)
        repo = os.getenv("QWEN_GGUF_REPO", DEFAULT_REPO)
        files = _files_for_model(os.getenv("TTS_MODEL", "qwen3-tts-0.6b"))
        missing = [
            filename
            for filename in files
            if not (models_dir / filename).is_file()
            or (models_dir / filename).stat().st_size == 0
        ]
        if missing:
            from huggingface_hub import hf_hub_download

            print(
                f"[models] downloading {len(missing)} file(s) from {repo}", flush=True
            )
            for filename in missing:
                print(f"[models] {filename}", flush=True)
                hf_hub_download(
                    repo_id=repo,
                    filename=filename,
                    local_dir=str(models_dir),
                )
            print("[models] ready", flush=True)

    argv = sys.argv[1:] or ["python", "main.py"]
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
