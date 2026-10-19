# Vendored code notice

The modules in this directory are trimmed copies of inference code from
[fishaudio/fish-speech](https://github.com/fishaudio/fish-speech), revision
`89474bbb3b` (the commit that added the OpenAudio S1 model structure), plus
the vector-quantizer primitives from
[descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec):

| File | Origin |
| --- | --- |
| `llama.py`, `tokenizer.py`, `content_sequence.py`, `text.py` | fish-speech, `fish_speech/` |
| `modded_dac.py`, `rvq.py`, `quantize.py`, `layers.py` | fish-speech `fish_speech/models/dac/` and descript-audio-codec `dac/nn/` |

Changes: removed training/LoRA/hydra/loguru/audiotools code, dead CLI entry
points and non-inference model variants; imports are relative. The KV cache
now returns only the populated prefix (and masks are sliced to match) instead
of attending over `max_seq_len` keys every decode step; this is mathematically
equivalent and roughly halves CPU latency.

Licensing: fish-speech is distributed under the Fish Audio Research License
(research and non-commercial use; commercial use needs a separate license),
and the S1-mini weights are CC-BY-NC-SA-4.0. descript-audio-codec is MIT.
