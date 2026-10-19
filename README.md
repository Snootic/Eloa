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
  fish_s1_mini/        FishAudio S1-mini adapter + vendored inference code
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
`reference_text`, and `overwrite`. It returns the exported file name and the
provider's refreshed voice list.

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

Quantization: there is no official quantized S1-mini checkpoint on the Hub,
so `options.quantize` applies torchao to the AR model at load time:
`"int8"` = dynamic activation+weight (falls back to `torch.ao`),
`"int8-weight"` = weight-only, which keeps the bf16 activations and requires
torchao. Both cut weight memory; neither is faster than bf16 on CPU at batch 1.
`options.device` defaults to `auto` (CUDA, then Intel XPU, then MPS, then
CPU) and also accepts an explicit `cuda`, `xpu`, `mps`, or `cpu`.
`options.dtype` accepts `bfloat16` (default), `float16`, or `float32`;
`options.checkpoint` can point at a local copy of the model directory or
another compatible Hub repo. Pre-quantized checkpoints are not accepted — the
int8 path quantizes the float weights after loading.

`options.compile = true` wraps the decode step in `torch.compile`. It targets
CUDA (where it is recommended); on CPU the first call spends minutes compiling
and steady-state throughput was slower than eager, so leave it off there.

CPU inference is substantially slower than realtime (the AR model is 860M
parameters); prefer a GPU, and use `options.quantize = "int8"` to halve weight
memory when RAM is tight. On a Fish-only CPU server, set
`options.threads` to the physical core count (measured ~1.5x faster than the
4-thread default); if Pocket TTS shares the process, 4 threads remains the
better compromise. `iterative_prompt` with a small `chunk_length` lowers
time-to-first-audio without changing throughput. On Intel Arc, install an XPU build of PyTorch
(`torch==2.9.1+xpu` from `download.pytorch.org/whl/xpu`); `device: "auto"`
then picks `xpu`, and masked attention automatically uses the math SDPA
kernel, which is the only one Intel's backend supports for that shape.

License: the weights are CC-BY-NC-SA-4.0 and the vendored fish-speech code is
under the Fish Audio Research License. Research and non-commercial use only;
commercial use requires a separate license from Fish Audio.

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
