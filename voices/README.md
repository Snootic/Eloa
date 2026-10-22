# Base voices

Eloa is shipped as a base voice for every provider that supports voice files.
Select `voice: "eloa"`; providers discover their own extension from this
directory.

| Provider | File(s) |
|---|---|
| Pocket TTS | `eloa.safetensors` |
| Chatterbox / Chatterbox Turbo (PyTorch) | `eloa.wav` + `eloa.lab` |
| Chatterbox Turbo ONNX | `eloa.wav` |
| Chatterbox Multilingual GGUF | `eloa.gguf` |
| Chatterbox Multilingual pt-BR GGUF | `eloa-ptbr.gguf` |
| Chatterbox Turbo GGUF | `eloa-turbo.gguf` |
| FishAudio S1-mini | `eloa.wav` + `eloa.lab` |
| F5-TTS | `eloa.wav` + `eloa.lab` |
| NeuTTS | `eloa.wav` + `eloa.lab` (encode to `.pt` when the provider is installed) |
| Qwen3-TTS | `eloa.spk` + `eloa.rvq` + `eloa.txt` |

Chatterbox Turbo is English-only; Eloa is included for completeness but its
Portuguese timbre/accent is not a Turbo-supported use case.

Kokoro does not support reference-audio cloning; it keeps its built-in voices.
