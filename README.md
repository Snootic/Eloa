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
  qwen3_tts.py         Qwen3-TTS via qwentts.cpp (Vulkan/CUDA/CPU)
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
`config`. `GET /v1/audio/providers` lists every registered provider, its
default voice, and the voices exported under `voices/`.

The WebSocket endpoint `ws://127.0.0.1:8082/v1/audio/speech/stream` takes the
same request body and replies with a `speech.start` event (format, sample
rate, channels), binary audio frames, then `speech.done`. PCM16 HTTP responses
carry `X-Sample-Rate` and `X-Audio-Format` headers; use them instead of
assuming 24 kHz.

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

## Qwen3-TTS (GGML / Vulkan)

`provider: "qwen3-tts"`, models `qwen3-tts-0.6b` (default) and
`qwen3-tts-1.7b`. This runs the C++ [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp)
port through GGML, so no PyTorch or OpenVINO model runs at request time. The
0.6B Base model clones a voice from a reference clip; its 1024-dim speaker
embedding is extracted once and cached as a `.spk` file.

Build the engine and fetch the two GGUFs:

```sh
git clone --recurse-submodules https://github.com/ServeurpersoCom/qwentts.cpp
cd qwentts.cpp && ./buildvulkan.sh   # or buildsycl.sh / buildcuda.sh / buildcpu.sh

mkdir -p models
hf download Serveurperso/Qwen3-TTS-GGUF \
  qwen-talker-0.6b-base-Q8_0.gguf \
  qwen-tokenizer-12hz-Q8_0.gguf --local-dir models
```

Point the provider at both binaries and the model files:

```json
{
  "provider": "qwen3-tts",
  "model": "qwen3-tts-0.6b",
  "voice": "larissa",
  "input": "Olá, esta é uma demonstração do Qwen3 TTS.",
  "options": {
    "talker": "/path/to/qwen-talker-0.6b-base-Q8_0.gguf",
    "codec": "/path/to/qwen-tokenizer-12hz-Q8_0.gguf",
    "binary": "/path/to/qwentts.cpp/build/qwen-tts",
    "codec_binary": "/path/to/qwentts.cpp/build/qwen-codec",
    "device": "Vulkan0",
    "language": "Portuguese"
  }
}
```

Supported options are `talker`, `codec`, `binary`, `codec_binary`, `device`,
`language`, `seed`, `temperature`, `top_k`, `top_p`, `repetition_penalty`,
`max_new_tokens`, `codec_chunk_dur`, and `timeout`. `language` defaults to
`auto`; the engine supports Chinese, English, Japanese, Korean, German, French,
Russian, Portuguese, Spanish, and Italian.

Reference clips can be any audio format: the provider resamples them to 24 kHz
with ffmpeg, runs `qwen-codec`, and caches the `.spk` embedding. Exporting a
voice writes the `.spk` into `voices/`, so later requests only need
`voice`/`options`. The 0.6B model needs about 1.3 GB of GGUF weights.

Measured here with the 0.6B Q8 model on an Arc B580: **5.3× realtime warm**
(13.0 s audio in 2.5 s), and **2.2× end-to-end through the provider** for a
6.3 s sentence including process and model startup. OpenVINO GenAI also has a
Qwen3-TTS pipeline, but it measured 0.5–0.8× on the same GPU, so it is not the
provider used here.

On the 12.8 GB card, 0.6B Q8 peaks at **2.2 GB** for a 13 s request and
**3.3 GB** for 80 s with `codec_chunk_dur: 4` (5.2 GB at the default 24 s).
1.7B Q8 peaks at **3.3 GB** for 12 s and **4.4 GB** for 80 s. Four concurrent
0.6B short requests fit (12.66 GB total); with `codec_chunk_dur: 4`, three
concurrent long 0.6B requests also fit (10.80 GB total). The 1.7B fits three
short or two long requests. These concurrency figures come from separate
processes — the engine serializes generation per model key.

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
