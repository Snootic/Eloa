# Pocket TTS server

A small HTTP/WebSocket text-to-speech server with a model-agnostic core.
Pocket TTS and FishAudio S1-mini ship as providers; any other backend can be
plugged in without touching the server, caching, voice handling, or audio
encoding.

## Layout

```
tts_core/              provider-agnostic core
  audio.py             AudioFormat + float32/PCM16 conversion
  provider.py          TTSProvider / SynthesisModel contracts
  registry.py          provider registry + `tts_core.providers` entry points
  encoding.py          ffmpeg-backed Ogg Opus encoding
  engine.py            model cache, voice cache, speech streams
providers/             officially supported providers
  pocket_tts.py        Pocket TTS adapter
  kokoro.py            Kokoro-82M, CPU-first with built-in voices
  neutts.py            NeuTTS Air/Nano (optional neutts package)
  qwen3_tts.py         Qwen3-TTS via qwentts.cpp (Vulkan/CUDA/CPU)
  qwen3_native.py      ctypes binding to the shared qwentts.cpp C ABI
  chatterbox_gguf.py   Chatterbox Multilingual via codec.cpp GGUF
  fish_s1_mini/        FishAudio S1-mini adapter + vendored inference code
  chatterbox.py        Chatterbox / Chatterbox-Turbo (optional package)
  f5_tts.py            F5-TTS (optional package)
server/                FastAPI HTTP + WebSocket surface
main.py                entrypoint
```

The core never imports `pocket_tts` or `torch`; providers hand it mono
float32 NumPy chunks and the core does caching, voice lookup, and encoding.

## Run

```sh
uv run main.py
```

Synthesis over HTTP:

```sh
curl -X POST http://127.0.0.1:8082/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "english", "voice": "camilly-english", "input": "Hello.", "response_format": "opus_24k"}' \
  --output speech.ogg
```

`provider` and `model` are optional and fall back to the engine defaults.
`options` carries provider-specific tuning; for Pocket TTS it accepts
`temp`, `lsd_decode_steps`, `noise_clamp`, `eos_threshold`, `quantize`, and
`config`. Deployment-wide provider options can be scoped with
`TTS_PROVIDER_OPTIONS={"qwen3-tts":{...}}`; only the selected provider receives
that map, so Qwen binary/model paths cannot leak into Pocket or Kokoro.
`GET /v1/audio/providers` lists every registered provider, its default voice,
and the voices exported under `voices/`.

The WebSocket endpoint `ws://127.0.0.1:8082/v1/audio/speech/stream` takes the
same request body and replies with a `speech.start` event (format, sample
rate, channels), binary audio frames, then `speech.done`. PCM16 HTTP responses
carry `X-Sample-Rate` and `X-Audio-Format` headers; use them instead of
assuming 24 kHz.

### Base voices

`voices/eloa.*` ships as a base voice. Select `voice: "eloa"`; providers
discover the format they support automatically: Pocket TTS `.safetensors`,
WAV-cloning providers `.wav` + `.lab`, Qwen `.spk` + `.rvq` + `.txt`, and
Chatterbox `.gguf`. See `voices/README.md` for the full matrix. Eloa is
available out of the box rather than replacing each provider's existing
default voice.

### Qwen3-TTS on Vulkan (workstation)

`run_qwen_vulkan.sh [model-alias]` starts Qwen3-TTS on an Intel Arc GPU. The
model alias selects the GGUF without editing talker/codec paths:

```sh
# test deployment: detached background process, no service manager
setsid nohup ./run_qwen_vulkan.sh > /tmp/qwen-tts.log 2>&1 < /dev/null &
```

Restart after source changes:

```sh
kill "$(ss -ltnp | sed -n 's/.*:8084 .*pid=\([0-9]*\).*/\1/p')"
setsid nohup ./run_qwen_vulkan.sh > /tmp/qwen-tts.log 2>&1 < /dev/null &
```

`TTS_PROVIDER_OPTIONS` keeps those paths under `qwen3-tts`; other providers
never receive them. Expose the local port to a Docker gateway on another host
with a Tailscale TCP proxy (`tailscale serve --tcp 18084 tcp://127.0.0.1:8084`).

### Test pages

Serve the repo and open the pages over HTTP (some browsers block
`file://` fetches to localhost):

```sh
python3 -m http.server 8083
# then open http://127.0.0.1:8083/test_http.html
```

| Page | Purpose |
| --- | --- |
| `test_http.html` | streaming HTTP synthesis, provider/model/voice/options selectors |
| `test_websocket.html` | the same over the WebSocket endpoint |
| `test_clone.html` | upload a reference clip, clone a voice, then speak it |

Provider, model, and voice are drop-downs populated from
`GET /v1/audio/providers`; the model list comes from each provider and the
voice list from `voices/` plus the engine/provider default. Query parameters
still prefill them, for example
`test_http.html?provider=fish-s1-mini&model=s1-mini&voice=ad-women&text=Hello`.

Clone a voice from the command line:

```sh
curl -F name=my-voice \
  -F provider=fish-s1-mini \
  -F reference_text="The words spoken in the clip" \
  -F overwrite=false \
  -F file=@voices/ad-women.wav \
  http://127.0.0.1:8082/v1/audio/voices/clone
```

The clone endpoint accepts `file` (32 MiB max), `name`, optional `provider`,
`reference_text`, `options` (a JSON object, matching the speech body), and
`overwrite`. It returns the exported file name and the provider's refreshed
voice list.

## Docker

Four images cover the execution backends:

| Environment file | Image contents | Host port |
|---|---|---:|
| `compose.cpu.yaml` | Pocket TTS + Kokoro, torch CPU | 8082 |
| `compose.openvino.yaml` | Chatterbox-Turbo ONNX, OpenVINO, Intel GPU runtime | 8083 |
| `compose.vulkan.yaml` | Qwen3-TTS via qwentts.cpp built with Vulkan | 8084 |
| `compose.cuda.yaml` | Qwen3-TTS via qwentts.cpp built with CUDA | 8085 |

`compose.yaml` holds the shared project name, external `local` network, and
cache volume. The environment files extend the common service template in
`compose.base.yaml`. Merge the shared file with one environment file:

```sh
docker compose -f compose.yaml -f compose.cpu.yaml up --build
docker compose -f compose.yaml -f compose.openvino.yaml up --build
docker compose -f compose.yaml -f compose.vulkan.yaml up --build
docker compose -f compose.yaml -f compose.cuda.yaml up --build
```

Use `-f compose.yaml -f compose.<env>.yaml down` to stop that environment.

All images use the same configurable entrypoint:

| Variable | Default |
|---|---|
| `TTS_PROVIDER` | `pocket-tts` |
| `TTS_MODEL` | `portuguese` |
| `TTS_VOICE` | `camilly-portuguese` |
| `TTS_VOICE_SOURCE` | unset |
| `TTS_OPTIONS` | `{}` (JSON object, applied to every provider) |
| `TTS_PROVIDER_OPTIONS` | `{}` (JSON map keyed by provider name) |
| `TTS_PROVIDER_OPTIONS_FILE` | unset (shared JSON map; env options win) |
| `TTS_MODEL_CATALOG` | `config/models.json` |
| `TTS_VOICES_DIR` | `voices` |
| `TTS_MAX_CACHED_MODELS` | `2` |
| `TTS_HOST` / `TTS_PORT` | `0.0.0.0` / `8082` |

The CPU image installs the `kokoro` extra by default. Build it slimmer with
`--build-arg UV_EXTRAS=` if Kokoro is not needed.

### GPU prerequisites

- **OpenVINO:** `/dev/dri`. The image is based on Intel's official OpenVINO
  runtime image, which ships the current Level Zero driver; the host kernel
  driver must be present.
- **Vulkan:** `/dev/dri`; Mesa's ANV driver is in the image.
- **CUDA:** NVIDIA driver + NVIDIA Container Toolkit; Compose requests all
  GPUs. The CUDA image is built from `nvidia/cuda:12.8.1`.

GPU containers run as root with `security_opt: label=disable` so they can open
`/dev/dri`; the CPU environment remains the non-GPU option.

SELinux hosts need the `:z` relabel that the Compose file already applies to
bind mounts. The GPU environments additionally disable SELinux labeling for device
access.

### Model and voice mounts

Model weights are **not** baked into the images. The base Compose template
mounts:

- `./voices` → `/app/voices`
- `./benchmarks` → `/app/benchmarks` (Helena ICL reference)
- `./models` → `/models` (Qwen GGUF)
- the `tts-hf-cache` volume → `/cache/huggingface`

For the Qwen environments, download the 0.6B Q8 files:

```sh
mkdir -p models
uvx --from huggingface_hub hf download Serveurperso/Qwen3-TTS-GGUF \
  qwen-talker-0.6b-base-Q8_0.gguf qwen-tokenizer-12hz-Q8_0.gguf --local-dir models
```

The Vulkan/CUDA defaults use the Helena voice in
`benchmarks/helena/helena-24k.wav` with its `.lab` transcript. If that folder
is absent, set `TTS_VOICE_SOURCE` and `TTS_VOICE` to a reference clip you have
(and provide its transcript), or send provider options per HTTP request.

The OpenVINO environment defaults to `chatterbox-turbo-ov` with
`voices/ad-women.wav`; set `TTS_PROVIDER=fish-s1-mini` etc. to use another
installed provider. Fish is gated and needs `HF_TOKEN`.

The Qwen images build `qwentts.cpp` at commit `5916bf7` with
`-DQWEN_SHARED=ON` and load `libqwen.so` natively. The Vulkan environment uses
`Vulkan0`; CUDA uses `CUDA0`. The images read model aliases from
`/app/config/models.json` and provider defaults from `/app/config/providers.json`,
so changing `TTS_MODEL` (for example to `qwen3-tts-0.6b-q6`) or the request
`model` switches quantizations without editing binary/model paths. Q4_K_M is
not recommended for the 0.6B talker; use Q8_0 or Q6_K.

## Kokoro (CPU)

`provider: "kokoro"`, `model: "kokoro-82m"`. Kokoro is 82M parameters,
Apache-2.0, and ships 54 named voices instead of reference clips. Install:

```sh
uv sync --extra kokoro
```

```sh
curl -X POST http://127.0.0.1:8082/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"provider": "kokoro", "voice": "af_heart", "input": "Hello from Kokoro.", "response_format": "opus_24k"}' \
  --output kokoro.ogg
```

Voices are prefixed by language: `af_*`/`am_*` English, `bf_*`/`bm_*` British,
`ef_*`/`em_*` Spanish, `ff_*` French, `hf_*`/`hm_*` Hindi, `if_*`/`im_*`
Italian, `jf_*`/`jm_*` Japanese, `pf_*`/`pm_*` Portuguese, `zf_*`/`zm_*`
Chinese. The provider creates the matching G2P pipeline per voice, so no
`lang_code` option is required. `options.speed`, `options.threads`, and
`options.device` (`cpu` default) are supported.

Measured here: **3.2–3.5× realtime on CPU with 4 threads** (3.3 s audio in
~1 s), including Portuguese. `GET /v1/audio/providers` lists all 54 voices.

## NeuTTS Air / Nano

`provider: "neutts"`, models `neutts-air` and `neutts-nano`. NeuTTS is a small
LLM + NeuCodec stack with instant voice cloning. `neutts` pins its own
Transformers/torch versions, so install it in a separate environment rather
than the default server venv:

```sh
uv venv ~/.venvs/neutts --python 3.12
uv pip install --python ~/.venvs/neutts/bin/python neutts
# Nano GGUF backbone:
uv pip install --python ~/.venvs/neutts/bin/python "neutts[llama]"

PYTHONPATH=. ~/.venvs/neutts/bin/python main.py
```

The official `neuphonic/neutts-air`, `neuphonic/neutts-nano`, and
`neuphonic/neucodec` repositories are gated on Hugging Face; accept their
terms and authenticate before use. Override `backbone_repo`/`codec_repo` to
use a mirror or local checkpoint.

A reference clip **and its exact transcript** are required: pass
`reference_text` to the clone endpoint, or place `<clip>.lab`/`<clip>.txt`
next to the reference. Exporting a voice writes a `.pt` code file and `.txt`.

Air is English, ~552M total parameters, Apache-2.0. Nano is English, ~229M,
under the NeuTTS Open License 1.0; German/French/Spanish variants are separate
repos. There is no official Portuguese Nano. Measured on CPU here: Air **0.24–
0.25× realtime**, Pocket TTS **2.2–2.9×** on the same clips. The official Nano
could not be benchmarked because its weights/codec are gated; details and the
Air/Pocket audio are in `benchmarks/neutts/`.

## Model aliases and GGUF quantization

`config/models.json` is the shared alias catalog used by the providers and
copied into the Docker images. Qwen aliases resolve to
`models/qwen-talker-{size}-base-{quant}.gguf` automatically:

| Alias | Talker |
|---|---|
| `qwen3-tts-0.6b` / `-q8` | 0.6B Q8_0 |
| `qwen3-tts-0.6b-q6` | 0.6B Q6_K |
| `qwen3-tts-0.6b-q5` | 0.6B Q5_K_M |
| `qwen3-tts-0.6b-q4` | 0.6B Q4_K_M |
| `qwen3-tts-1.7b*` | 1.7B Q8/Q6/Q5/Q4 |

Set `TTS_MODEL=qwen3-tts-0.6b-q6` (or send `"model"` in the request) to switch
quantization without changing binary, codec, or voice. `models_dir` points at
the GGUF directory; explicit `talker`/`codec` options still override the alias.
`Q4_K_M` is listed for completeness but is broken for the 0.6B talker.

NeuTTS exposes `neutts-air-q4/q8` and `neutts-nano-q4/q8` aliases; those
repositories are gated.

Exported Qwen voices include `.spk`, `.rvq`, `.txt`, and `.ref.wav`; switching
between 0.6B and 1.7B regenerates the size-specific speaker embedding from the
reference clip automatically.

`config/providers.json` holds deployment-wide provider options. Use
`TTS_PROVIDER_OPTIONS_FILE=/app/config/providers.json` plus a small
`TTS_PROVIDER_OPTIONS` override for the device.

## Qwen3-TTS (GGML / Vulkan)

`provider: "qwen3-tts"`, models `qwen3-tts-0.6b` (default) and
`qwen3-tts-1.7b`. This runs the C++ [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp)
port through GGML, so no PyTorch or OpenVINO model runs at request time. The
0.6B Base model clones a voice from a reference clip; its 1024-dim speaker
embedding is extracted once and cached as a `.spk` file.

Build the engine, shared library, and CLI tools, then fetch the two GGUFs:

```sh
git clone --recurse-submodules https://github.com/ServeurpersoCom/qwentts.cpp
cd qwentts.cpp
cmake -S . -B build-shared -DGGML_VULKAN=ON -DQWEN_SHARED=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-shared -j"$(nproc)" --target qwen qwen-tts qwen-codec

mkdir -p models
hf download Serveurperso/Qwen3-TTS-GGUF \
  qwen-talker-0.6b-base-Q8_0.gguf \
  qwen-tokenizer-12hz-Q8_0.gguf --local-dir models
```

Use `-DGGML_CUDA=ON` for CUDA or `-DGGML_VULKAN=OFF` for CPU. The recommended
runtime loads `build-shared/libqwen.so` in-process and keeps the model resident.
The CLI binaries remain the fallback and are still used by `qwen-codec` to
encode new reference voices.

Point the provider at the shared library and model files:

```json
{
  "provider": "qwen3-tts",
  "model": "qwen3-tts-0.6b",
  "voice": "larissa",
  "input": "Olá, esta é uma demonstração do Qwen3 TTS.",
  "options": {
    "talker": "/path/to/qwen-talker-0.6b-base-Q8_0.gguf",
    "codec": "/path/to/qwen-tokenizer-12hz-Q8_0.gguf",
    "native_library": "/path/to/qwentts.cpp/build-shared/libqwen.so",
    "backend": "native",
    "binary": "/path/to/qwentts.cpp/build/qwen-tts",
    "codec_binary": "/path/to/qwentts.cpp/build/qwen-codec",
    "device": "Vulkan0",
    "language": "Portuguese"
  }
}
```

Supported options are `talker`, `codec`, `models_dir`, `native_library`,
`backend` (`auto`, `native`, `cli`), `use_fa`, `clamp_fp16`, `max_batch`,
`binary`, `codec_binary`, `device`, `language`, `seed`, `temperature`, `top_k`,
`top_p`, `repetition_penalty`, `max_new_tokens`, `codec_chunk_dur`, and
`timeout`. With `backend: "auto"`, setting `native_library` selects the shared
library; without it the provider falls back to the `qwen-tts` subprocess. If
the sibling GGML libraries are not on the loader path, set
`LD_LIBRARY_PATH=/path/to/qwentts.cpp/build-shared`.
`language` defaults to `auto`; the engine supports Chinese, English, Japanese,
Korean, German, French, Russian, Portuguese, Spanish, and Italian.

Reference clips can be any audio format: the provider resamples them to 24 kHz
with ffmpeg, runs `qwen-codec`, and caches the `.spk` embedding plus `.rvq`
codes. Exporting a voice writes `.spk`, `.rvq`, and `.txt` into `voices/`, so
later requests only need `voice`/`options`. The 0.6B model needs about 1.3 GB
of GGUF weights.

**Supply a reference transcript.** Voice cloning with the speaker embedding
alone (x-vector) did not preserve the test voice: Q4_K_M 1.7B produced a
112 Hz male voice and Q8_0 0.6B a 197 Hz voice, versus 264 Hz for the
reference. ICL with `.rvq` + the reference transcript moved them to 209 Hz and
238 Hz and produced the correct speaker. The provider uses ICL automatically
when the voice has a `.rvq` and a `.txt`/`.lab` sidecar. Over HTTP, send
`reference_text` to the clone endpoint; from Python, place `<clip>.lab` next
to the reference or `<name>.txt` next to `voices/<name>.spk`.

Qwen3-TTS exposes only a single `portuguese` language token, with no separate
pt-BR mode. ICL conditions on the reference and transcript, which is the best
available lever for Brazilian accent, but the model cannot guarantee pt-BR
phonology; judge that by listening.

Measured here with the 0.6B Q8 native backend on an Arc B580: the first
request after model load reaches the first audio chunk in **0.25 s** and
synthesizes 3.9 s of audio in **1.04 s**; a warm request on the resident handle
reaches first audio in **0.055 s** and finishes in **0.82 s** (about **4.8×
realtime**). The subprocess fallback starts a fresh engine per request and is
substantially slower. OpenVINO GenAI also has a Qwen3-TTS pipeline, but it
measured 0.5–0.8× on the same GPU, so it is not the provider used here.

On the 12.8 GB card, 0.6B Q8 peaks at **2.2 GB** for a 13 s request and
**3.3 GB** for 80 s with `codec_chunk_dur: 4` (5.2 GB at the default 24 s).
1.7B Q8 peaks at **3.3 GB** for 12 s and **4.4 GB** for 80 s. Four concurrent
0.6B short requests fit (12.66 GB total); with `codec_chunk_dur: 4`, three
concurrent long 0.6B requests also fit (10.80 GB total). The 1.7B fits three
short or two long requests. Those figures were measured with separate CLI
processes; the native backend keeps one resident context per model key and
serializes generation through it, so treat them as process-level upper bounds
rather than native concurrency limits.

### Quantization

The native context keeps the model and its KV cache resident. A 0.6B Q8
context allocates about 2.08 GB at load and 2.30 GB after synthesis; switching
to Q6 and back reloads the context. Keep `TTS_MAX_CACHED_MODELS=1` for Qwen
(the workstation script and Vulkan/CUDA images set it): caching two native
quantizations simultaneously costs about 4.3 GB, because each has its own
weights and KV cache. The old CLI fallback did not retain the previous
quantization and therefore peaked lower.

Any GGUF quantization works by pointing `talker` and `codec` at the matching
files; no separate engine path is needed. Measured with this runtime:

- **Q4_K_M 0.6B talker is not usable.** Whisper transcribes the output as
  `[música]`, including when paired with the Q8 tokenizer. The Q4 tokenizer
  itself is fine: Q8 talker + Q4 tokenizer matches Q8 output.
- **Q4_K_M 1.7B is intelligible**, but peaks at 3.26 GB for 13 s and 3.51 GB
  for 24 s.
- **Q8_0 0.6B is the quality/VRAM sweet spot**: 2.22 GB for 13 s and
  3.11–3.32 GB for 17–26 s. For a hard 3 GB *process* ceiling, keep utterances
  around 13 s; a 20–30 s request is 0.1–0.3 GB over. The OS/desktop also adds
  roughly 1–1.7 GB to total GPU usage, so 3 GB *total* is not achievable at
  these contexts.

`codec_chunk_dur: 4` is the speed/VRAM default recommendation; `1` saves about
0.2 GB at 26 s but is slower. These are ASR checks, not listening tests.

## FishAudio S1-mini

`provider: "fish-s1-mini"` loads [`fishaudio/s1-mini`](https://huggingface.co/fishaudio/s1-mini)
(0.5B dual-AR model + 44.1 kHz firefly codec). The repo is gated: accept its
terms on the Hub, then provide `HF_TOKEN` or a stored token. Weights download
on first use into the Hugging Face cache (about 3.6 GB).

```json
{
  "provider": "fish-s1-mini",
  "model": "s1-mini",
  "voice": "ad-women",
  "input": "Hello from FishAudio.",
  "options": {
    "quantize": "int8",
    "temperature": 0.8,
    "top_p": 0.8,
    "repetition_penalty": 1.1
  }
}
```

Voices are reference clips in `voices/<name>.wav` (any ffmpeg-readable audio
works; only the first 30 s are used). A `<name>.lab` sidecar supplies the
transcript of the clip; without it, set `options.reference_text`. If no
reference exists, the provider's `default` voice synthesizes an unconditioned
(random-timbre) voice.

Backend: `options.backend` is `auto` (default), `torch`, or `openvino`.
`auto` selects OpenVINO when it is installed and an Intel GPU is visible,
otherwise the PyTorch path. OpenVINO compiles the slow and fast transformer
steps into GPU graphs with explicit KV caches; sampling stays in Python.
`options.openvino_device` is `AUTO`, `GPU`, or `CPU`. Install it with
`uv sync --extra openvino` (or `pip install openvino`). Intel GPUs require
OpenVINO to initialize before torch, so the provider creates `ov.Core()`
first — if you embed the server, import `providers` before torch. The codec
still runs on the torch device. PyTorch XPU support was removed. On an Arc
B580, a 1.8 s utterance took about 7.3 s wall time (prefill 0.3 s,
autoregressive decoding 4.3 s, codec 1.8 s).

Torch path (also used for CUDA/MPS/CPU): `options.device` defaults to `auto`
(CUDA, then MPS, then CPU) and accepts `cuda`, `mps`, or `cpu`.
`options.dtype` accepts `bfloat16` (default), `float16`, or `float32`.
`options.checkpoint` can point at a local copy of the model directory or
another compatible Hub repo. Pre-quantized checkpoints are not accepted.

Quantization (torch backend, no official quantized checkpoint on the Hub):
`options.quantize` applies torchao at load time — `"int8"` = dynamic
activation+weight (falls back to `torch.ao`), `"int8-weight"` = weight-only,
which keeps bf16 activations and requires torchao. Both cut weight memory;
neither was faster than bf16 on CPU at batch 1. `options.compile = true`
wraps the torch decode step in `torch.compile`; it targets CUDA, and on CPU
the first call spends minutes compiling and ran slower than eager.

CPU inference is substantially slower than realtime (the AR model is 860M
parameters). On a Fish-only CPU server, set `options.threads` to the physical
core count (measured ~1.5x faster than the 4-thread default); if Pocket TTS
shares the process, 4 threads remains the better compromise.
`iterative_prompt` with a small `chunk_length` lowers time-to-first-audio
without changing throughput.

License: the weights are CC-BY-NC-SA-4.0 and the vendored fish-speech code is
under the Fish Audio Research License. Research and non-commercial use only;
commercial use requires a separate license from Fish Audio.

## Chatterbox Multilingual GGUF (native)

`provider: "chatterbox-gguf"` loads the [BricksDisplay Chatterbox
Multilingual GGUFs](https://huggingface.co/BricksDisplay/Chatterbox-Multilingual-TTS-GGUF)
through [codec.cpp](https://github.com/mybigday/codec.cpp). The Python binding
keeps the T3 backbone and codec resident in-process; the codec runs on the
same GGUF backends as the Qwen provider (Vulkan, CUDA, or CPU).

Build codec.cpp with shared libraries, then fetch the Q8 or F16 pair:

```sh
git clone --recursive https://github.com/mybigday/codec.cpp
cd codec.cpp
cmake -S . -B build -DBUILD_SHARED_LIBS=ON -DCODEC_TTS_SERVER=OFF \
  -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release       # or -DGGML_VULKAN=ON
cmake --build build -j --target codec_shared ttsbackbone

cd ..
hf download BricksDisplay/Chatterbox-Multilingual-TTS-GGUF \
  chatterbox-mtl-q8_0.gguf codec-q8_0.gguf --local-dir models/chatterbox
```

Provider options:

```json
{
  "models_dir": "/path/to/models/chatterbox",
  "quant": "q8_0",
  "use_gpu": true,
  "language": "pt",
  "segment_bytes": 100,
  "emotion": 0.5,
  "flow_steps": 4,
  "python_path": "/path/to/codec.cpp/python",
  "library": "/path/to/codec.cpp/build/libcodec.so",
  "backbone_library": "/path/to/codec.cpp/build/ttsbackbone/libttsbackbone.so"
}
```

Aliases are `chatterbox-mtl`, `chatterbox-mtl-q8_0`, `chatterbox-mtl-q4_k_m`,
and `chatterbox-mtl-f16`. Use `chatterbox-mtl-f16` for the full model.
Reference voices are saved as `voices/<name>.gguf`; `POST
/v1/audio/voices/clone` works with the same provider.

`segment_bytes` controls the latency/quality tradeoff: Chatterbox emits a
finished segment at a time and has no intra-segment streaming. `emotion` maps
to Chatterbox's `emotion_adv` scalar (0–1.5); changing it reloads the model.
`flow_steps` lowers decoder cost (6 is the model default, 4 is a good
tradeoff).

Measured on Arc B580 Vulkan, Helena voice, Portuguese 9.4 s text, 4 flow
steps:

| Model | First audio | Total | Audio | Peak VRAM |
|---|---:|---:|---:|---:|
| Q8, whole utterance | 2.44 s | 2.44 s | 9.4 s | 1.26 GB |
| Q8, `segment_bytes=100` | **1.11 s** | 3.55 s | 9.7 s | 1.26 GB |
| F16, whole utterance | 2.74 s | 2.74 s | 9.4 s | **1.72 GB** |
| F16, `segment_bytes=100` | 1.20 s | 3.86 s | 9.2 s | 1.72 GB |

The first F16 request after loading can be ~3.7 s while shaders warm. On a
GTX 1650 D6 (4 GB, 192 GB/s, 896 CUDA cores) expect roughly **2–3× the B580
times before CUDA-specific gains**: about 4–8 s for the Q8 9.4 s utterance
and 2–4 s TTFA with `segment_bytes=100`. Q8's 1.26 GB and F16's 1.72 GB both
fit in 4 GB, but leave limited headroom for the desktop. Prefer Q8 on a 4 GB
card; use the CUDA build.

`tts-server`/the built-in Qwen path remain faster and stream earlier; the
Chatterbox native path is for emotion, expressiveness, and its cloning voice.

Emotion presets (sad, neutral, happy, enthusiastic, angry) are benchmarked
in [`benchmarks/chatterbox-emotions/results.md`](benchmarks/chatterbox-emotions/results.md),
including generated audio and Q8/F16 VRAM numbers.

The Brazilian-Portuguese fine-tune (`ResembleAI/Chatterbox-Multilingual-pt-br`)
was converted to GGUF and benchmarked in
[`benchmarks/chatterbox-emotions/controlled/ptbr-results.md`](benchmarks/chatterbox-emotions/controlled/ptbr-results.md).
Its local service is `run_chatterbox_ptbr_vulkan.sh` on port 8087; the
generic multilingual service remains on 8086.

## Optional providers: Chatterbox and F5-TTS

Three more providers are included but not installed by default: the torch
packages pin incompatible torch/numpy stacks.

| Provider | Models | Runtime | Reference audio | License |
| --- | --- | --- | --- | --- |
| `chatterbox-turbo-ov` | `chatterbox-turbo` | OpenVINO (torch-free) | audio prompt only | MIT |
| `chatterbox` | `chatterbox`, `chatterbox-turbo` | torch | audio prompt only | MIT |
| `f5-tts` | `F5TTS_v1_Base`, `F5TTS_Base`, `E2TTS_Base` | torch | audio + transcript | code MIT, weights CC-BY-NC-4.0 |

The ONNX Turbo provider needs no torch and can live in the default
environment:

```sh
uv sync --extra chatterbox-onnx
```

It downloads the official fp16 ONNX graphs (~1.7 GB) on first use and copies
them out of the Hugging Face symlink layout so OpenVINO can read the external
weights. The language model and embeddings run on the OpenVINO GPU; the speech
encoder and decoder run on CPU because their fp16 graphs are not usable on the
current Intel GPU driver.

Install the torch providers each in its own environment (they pin torch):

```sh
uv venv ~/.venvs/chatterbox --python 3.12
uv pip install --python ~/.venvs/chatterbox/bin/python -e .
uv pip install --python ~/.venvs/chatterbox/bin/python chatterbox-tts
PYTHONPATH=. ~/.venvs/chatterbox/bin/python main.py

uv venv ~/.venvs/f5 --python 3.12
uv pip install --python ~/.venvs/f5/bin/python -e .
uv pip install --python ~/.venvs/f5/bin/python f5-tts
PYTHONPATH=. ~/.venvs/f5/bin/python main.py
```

Installing the provider last lets it pin torch; the project's other providers
may then run with that torch version instead. Keep the default `.venv`
unchanged if you only need Pocket/Fish.

The registry skips a provider whose package is missing, so the default
environment keeps working. F5-TTS requires the reference transcript: put a
`<voice>.lab` sidecar next to the clip or pass `options.reference_text`.
Chatterbox uses the clip itself; its transcript is ignored.

Measured on this workstation (~2.5 s utterances; CPU rows use 4 threads):

| Model | Device | Speed |
| --- | --- | ---: |
| Pocket TTS | CPU | 2.8× realtime |
| Kokoro-82M | CPU | 3.2–3.5× realtime |
| Qwen3-TTS 0.6B | Arc B580, Vulkan | 5.3× warm; 2.2× via provider |
| **Chatterbox-Turbo (ONNX)** | **Arc B580, OpenVINO** | **0.79–0.84×** |
| FishAudio S1-mini | Arc B580, OpenVINO | 0.25× |
| Chatterbox-Turbo | CPU | 0.19× |
| F5-TTS, 8 steps | CPU | 0.10× |
| Chatterbox (original) | CPU | 0.07× |
| F5-TTS, 16 steps | CPU | 0.04× |

Chatterbox-Turbo on OpenVINO is the best non-Pocket result: ~4× its CPU speed
and ~3× faster than Fish on the same GPU. Sampling is required; greedy decoding
degenerates into a loop. The ONNX AR loop runs on GPU, while the one-time voice
encoder (~3 s) and per-utterance decoder (~0.7–1.0 s) remain on CPU. F5-TTS has
no ONNX export, so it was not tested on the Intel GPU; it remains a torch
CPU/CUDA/MPS provider.

These are speed measurements only — quality was not evaluated by listening.

## Add a provider

1. Implement a `SynthesisModel` (sample rate, voice conditioning, streaming
   synthesis) and a `TTSProvider` (`load_model`, optional `resolve_options`)
   from `tts_core.provider`.
2. Register it in `providers/__init__.py::register_builtin_providers`, or
   publish it from a separate package with an entry point:

   ```toml
   [project.entry-points."tts_core.providers"]
   my-tts = "my_package:MyTTSProvider"
   ```

3. Call it with `{"provider": "my-tts", "model": "..."}`.

Providers own their own dependencies: importing `tts_core` does not import
any model runtime.
