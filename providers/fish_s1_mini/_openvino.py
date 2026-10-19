"""OpenVINO backend for the FishAudio S1-mini autoregressive loop.

The slow and fast transformers are compiled as stateless OpenVINO graphs with
their KV caches as explicit inputs/outputs. Sampling stays in Python. The
`ov.Core` is created before torch is imported: on Intel GPUs, OpenVINO cannot
see the device once torch's XPU stack has initialized it.
"""

import logging
from collections.abc import Iterator

import numpy as np
import openvino as ov

logger = logging.getLogger(__name__)

CORE = ov.Core()

import torch
from torch import nn

from ._upstream.content_sequence import ContentSequence, TextPart, VQPart
from ._upstream.inference import _logits_to_probs, _multinomial_sample_one_no_sync
from ._upstream.llama import DualARTransformer, apply_rotary_emb
from ._upstream.text import split_text
from ._upstream.tokenizer import IM_END_TOKEN

NEG_INF = -1.0e9


def available_devices() -> list[str]:
    return list(CORE.available_devices)


def has_gpu() -> bool:
    return "GPU" in CORE.available_devices


def _attention(q, k, v, mask):
    scale = 1.0 / (q.shape[-1] ** 0.5)
    weights = (q @ k.transpose(-2, -1)) * scale + mask
    return torch.softmax(weights, dim=-1) @ v


def _apply_layer(layer, x, freqs, mask, k_cache, v_cache, *, pos=None):
    attn = layer.attention
    q_size = attn.n_head * attn.head_dim
    kv_size = attn.n_local_heads * attn.head_dim
    q, k_new, v_new = attn.wqkv(layer.attention_norm(x)).split(
        [q_size, kv_size, kv_size], dim=-1
    )
    q = q.view(1, 1, attn.n_head, attn.head_dim)
    k_new = k_new.view(1, 1, attn.n_local_heads, attn.head_dim).transpose(1, 2)
    v_new = v_new.view(1, 1, attn.n_local_heads, attn.head_dim).transpose(1, 2)

    if attn.attention_qk_norm:
        q = attn.q_norm(q)
        k_new = attn.k_norm(k_new.transpose(1, 2)).transpose(1, 2)

    q = apply_rotary_emb(q, freqs).transpose(1, 2)
    k_new = apply_rotary_emb(k_new.transpose(1, 2), freqs).transpose(1, 2)

    if pos is None:
        # Slow transformer: the cache grows by one token per step.
        k_full = torch.cat([k_cache, k_new], dim=2)
        v_full = torch.cat([v_cache, v_new], dim=2)
    else:
        # Fast transformer: fixed capacity, overwrite the slot for `pos`.
        positions = torch.arange(k_cache.shape[2], device=k_cache.device).view(
            1, 1, -1, 1
        )
        write = positions == pos.view(1, 1, 1, 1)
        k_full = torch.where(write, k_new, k_cache)
        v_full = torch.where(write, v_new, v_cache)

    k = k_full.repeat_interleave(attn.n_head // attn.n_local_heads, dim=1)
    v = v_full.repeat_interleave(attn.n_head // attn.n_local_heads, dim=1)
    y = _attention(q, k, v, mask)
    y = y.transpose(1, 2).contiguous().view(1, 1, -1)
    h = x + attn.wo(y)
    h = h + layer.feed_forward(layer.ffn_norm(h))
    return h, k_full, v_full


class _SlowStep(nn.Module):
    """Embedding + 28 slow layers + logits for one token, caches external."""

    def __init__(self, model: DualARTransformer):
        super().__init__()
        self.model = model
        self.num_codebooks = model.config.num_codebooks
        self.codebook_size = model.config.codebook_size

    def forward(self, tokens, freqs, mask, vq_mask, *caches):
        inp = tokens[:, :, 0]
        embeds = [
            self.model.codebook_embeddings(
                inp[:, index + 1] + index * self.codebook_size
            )
            for index in range(self.num_codebooks)
        ]
        vq = torch.stack(embeds, dim=1).sum(dim=1) * vq_mask
        x = (self.model.embeddings(inp[:, 0]) + vq)[:, None, :]

        new_caches = []
        for index, layer in enumerate(self.model.layers):
            x, k, v = _apply_layer(
                layer, x, freqs, mask, caches[2 * index], caches[2 * index + 1]
            )
            new_caches.extend([k, v])

        slow_out = self.model.norm(x)
        if self.model.config.tie_word_embeddings:
            logits = torch.nn.functional.linear(slow_out, self.model.embeddings.weight)
        else:
            logits = self.model.output(slow_out)
        return (logits, self.model.fast_project_in(x), *new_caches)


def _apply_prefill_layer(layer, x, freqs, mask):
    attn = layer.attention
    seq = x.shape[1]
    q_size = attn.n_head * attn.head_dim
    kv_size = attn.n_local_heads * attn.head_dim
    q, k, v = attn.wqkv(layer.attention_norm(x)).split(
        [q_size, kv_size, kv_size], dim=-1
    )
    q = q.view(1, seq, attn.n_head, attn.head_dim)
    k = k.view(1, seq, attn.n_local_heads, attn.head_dim)
    v = v.view(1, seq, attn.n_local_heads, attn.head_dim)

    if attn.attention_qk_norm:
        q = attn.q_norm(q)
        k = attn.k_norm(k)

    q = apply_rotary_emb(q, freqs)
    k_rope = apply_rotary_emb(k, freqs)

    q = q.transpose(1, 2)
    k_rope = k_rope.transpose(1, 2)
    v = v.transpose(1, 2)
    k_rep = k_rope.repeat_interleave(attn.n_head // attn.n_local_heads, dim=1)
    v_rep = v.repeat_interleave(attn.n_head // attn.n_local_heads, dim=1)
    y = _attention(q, k_rep, v_rep, mask)
    y = y.transpose(1, 2).reshape(1, seq, -1)
    h = x + attn.wo(y)
    h = h + layer.feed_forward(layer.ffn_norm(h))
    return h, k_rope, v


class _SlowPrefill(nn.Module):
    """Embedding + 28 slow layers over the whole prompt, returning KV caches."""

    def __init__(self, model: DualARTransformer):
        super().__init__()
        self.model = model
        self.num_codebooks = model.config.num_codebooks
        self.codebook_size = model.config.codebook_size

    def forward(self, tokens, freqs, mask, vq_mask):
        embeds = [
            self.model.codebook_embeddings(
                tokens[:, index + 1, :] + index * self.codebook_size
            )
            for index in range(self.num_codebooks)
        ]
        vq = torch.stack(embeds, dim=1).sum(dim=1) * vq_mask
        x = self.model.embeddings(tokens[:, 0, :]) + vq

        keys, values = [], []
        for layer in self.model.layers:
            x, k, v = _apply_prefill_layer(layer, x, freqs, mask)
            keys.append(k)
            values.append(v)

        last = x[:, -1:, :]
        slow_out = self.model.norm(last)
        if self.model.config.tie_word_embeddings:
            logits = torch.nn.functional.linear(slow_out, self.model.embeddings.weight)
        else:
            logits = self.model.output(slow_out)
        # One stacked tensor per cache side keeps the OpenVINO graph to four
        # outputs; 56 separate outputs crashed the frontend.
        return (
            logits,
            self.model.fast_project_in(last),
            torch.stack(keys, dim=0),
            torch.stack(values, dim=0),
        )


class _FastStep(nn.Module):
    """Projected hidden + 4 fast layers + codebook logits, caches external."""

    def __init__(self, model: DualARTransformer):
        super().__init__()
        self.model = model

    def forward(self, x, freqs, mask, pos, *caches):
        new_caches = []
        for index, layer in enumerate(self.model.fast_layers):
            x, k, v = _apply_layer(
                layer,
                x,
                freqs,
                mask,
                caches[2 * index],
                caches[2 * index + 1],
                pos=pos,
            )
            new_caches.extend([k, v])
        logits = self.model.fast_output(self.model.fast_norm(x))
        return (logits, *new_caches)


class OVGenerator:
    """Runs the Fish AR loop on an OpenVINO device."""

    def __init__(self, model: DualARTransformer, device: str = "AUTO"):
        self.model = model
        self.config = model.config
        self.tokenizer = model.tokenizer
        self.device = self._pick_device(device)
        self.semantic_begin = self.tokenizer.semantic_begin_id
        self.eos_id = self.tokenizer.get_token_id(IM_END_TOKEN)
        self.semantic_ids = set(self.tokenizer.semantic_id_to_token_id.values())
        self.fast_embedding = (
            model.fast_embeddings.weight.detach().to("cpu", torch.float32).numpy()
        )
        self.slow_caches_shape = (
            1,
            self.config.n_local_heads,
            1,
            self.config.head_dim,
        )
        self.fast_caches_shape = (
            1,
            self.config.fast_n_local_heads,
            self.config.num_codebooks,
            self.config.fast_head_dim,
        )
        self._build()

    def _pick_device(self, device: str) -> str:
        devices = available_devices()
        requested = device.upper()
        if requested == "AUTO":
            if "GPU" in devices:
                return "GPU"
            if "CPU" in devices:
                return "CPU"
            raise RuntimeError("OpenVINO has no usable devices")
        if requested not in devices:
            raise ValueError(
                f"OpenVINO device {device!r} is not available; have {devices}"
            )
        return requested

    def _build(self) -> None:
        logger.info("Compiling Fish OpenVINO graphs on %s", self.device)
        prefill = _SlowPrefill(self.model).eval()
        prefill_example = (
            np.zeros((1, self.config.num_codebooks + 1, 8), dtype=np.int64),
            np.zeros((1, 8, self.config.head_dim // 2, 2), dtype=np.float32),
            np.zeros((1, 1, 8, 8), dtype=np.float32),
            np.ones((1, 8, 1), dtype=np.float32),
        )
        self.prefill = CORE.compile_model(
            ov.convert_model(prefill, example_input=prefill_example), self.device
        )

        slow = _SlowStep(self.model).eval()
        slow_example = (
            np.zeros((1, self.config.num_codebooks + 1, 1), dtype=np.int64),
            np.zeros((1, self.config.head_dim // 2, 2), dtype=np.float32),
            np.zeros((1, 1, 1, 2), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
            *[
                np.zeros(self.slow_caches_shape, dtype=np.float32)
                for _ in range(2 * len(self.model.layers))
            ],
        )
        self.slow = CORE.compile_model(
            ov.convert_model(slow, example_input=slow_example), self.device
        )

        fast = _FastStep(self.model).eval()
        fast_example = (
            np.zeros((1, 1, self.config.fast_dim), dtype=np.float32),
            np.zeros((1, self.config.fast_head_dim // 2, 2), dtype=np.float32),
            np.zeros((1, 1, 1, self.config.num_codebooks), dtype=np.float32),
            np.array(0, dtype=np.int64),
            *[
                np.zeros(self.fast_caches_shape, dtype=np.float32)
                for _ in range(2 * len(self.model.fast_layers))
            ],
        )
        self.fast = CORE.compile_model(
            ov.convert_model(fast, example_input=fast_example), self.device
        )
        logger.info("Fish OpenVINO graphs ready")

    def _fast_caches(self) -> list[np.ndarray]:
        return [
            np.zeros(self.fast_caches_shape, dtype=np.float32)
            for _ in range(2 * len(self.model.fast_layers))
        ]

    def _mask(self, length: int, allowed_until: int, first_allowed: int = 0):
        mask = np.full((1, 1, 1, length), NEG_INF, dtype=np.float32)
        mask[..., first_allowed : allowed_until + 1] = 0.0
        return mask

    def _slow_prefill(self, prompt: np.ndarray):
        length = prompt.shape[1]
        tokens = prompt[None, :, :]
        freqs = self.model.freqs_cis[:length].float().cpu().numpy()[None, :, :, :]
        mask = np.triu(np.full((1, 1, length, length), NEG_INF, dtype=np.float32), k=1)
        vq_mask = np.array(
            [
                [
                    [1.0 if int(token) in self.semantic_ids else 0.0]
                    for token in prompt[0]
                ]
            ],
            dtype=np.float32,
        )
        result = self.prefill([tokens, freqs, mask, vq_mask])
        logits, hidden, keys, values = list(result.values())
        caches = []
        for index in range(len(self.model.layers)):
            caches.extend([keys[index], values[index]])
        return logits, hidden, caches

    def _slow_step(
        self,
        tokens: np.ndarray,
        pos: int,
        caches: list[np.ndarray],
        vq_mask: float,
        mask_length: int,
        allowed_until: int,
    ):
        freqs = self.model.freqs_cis[pos : pos + 1].float().cpu().numpy()
        mask = self._mask(mask_length, allowed_until)
        result = self.slow(
            [
                tokens,
                freqs,
                mask,
                np.array([[vq_mask]], dtype=np.float32),
                *caches,
            ]
        )
        outputs = list(result.values())
        return outputs[0], outputs[1], outputs[2:]

    def _fast_step(self, x, pos, caches):
        freqs = self.model.fast_freqs_cis[pos : pos + 1].float().cpu().numpy()
        mask = self._mask(self.config.num_codebooks, pos, first_allowed=0)
        result = self.fast(
            [
                x,
                freqs,
                mask,
                np.array(pos, dtype=np.int64),
                *caches,
            ]
        )
        outputs = list(result.values())
        return outputs[0], outputs[1:]

    def _sample(self, logits, previous, temperature, top_p, repetition) -> int:
        tensor = torch.from_numpy(np.asarray(logits, dtype=np.float32))
        probs = _logits_to_probs(
            tensor,
            previous_tokens=previous,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition,
        )
        return int(_multinomial_sample_one_no_sync(probs).item())

    def generate(
        self,
        *,
        text: str,
        prompt_text: str | None,
        prompt_tokens: torch.Tensor | None,
        max_new_tokens: int,
        top_p: float,
        repetition_penalty: float,
        temperature: float,
        chunk_length: int,
        iterative_prompt: bool,
        seed: int | None,
    ) -> Iterator[torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)

        base = ContentSequence(modality="interleave")
        if prompt_text is not None and prompt_tokens is not None:
            base.append(
                [TextPart(text=prompt_text), VQPart(codes=prompt_tokens)],
                add_end=True,
            )
        encoded_prompt = base.encode_for_inference(
            self.tokenizer, num_codebooks=self.config.num_codebooks
        ).numpy()

        texts = split_text(text, chunk_length) if iterative_prompt else [text]
        if not texts:
            raise ValueError("Text contains no speakable content")

        temperature_t = torch.tensor(temperature, dtype=torch.float32)
        top_p_t = torch.tensor(top_p, dtype=torch.float32)
        repetition_t = torch.tensor(repetition_penalty, dtype=torch.float32)

        for segment in texts:
            encoded = (
                ContentSequence()
                .append(TextPart(text=segment))
                .encode_for_inference(
                    self.tokenizer, num_codebooks=self.config.num_codebooks
                )
                .numpy()
            )
            prompt = np.concatenate([encoded_prompt, encoded], axis=1)
            codes = self._generate_segment(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature_t,
                top_p=top_p_t,
                repetition=repetition_t,
            )
            if codes.size:
                yield torch.from_numpy(codes)

    def _generate_segment(
        self,
        prompt: np.ndarray,
        *,
        max_new_tokens: int,
        temperature: torch.Tensor,
        top_p: torch.Tensor,
        repetition: torch.Tensor,
    ) -> np.ndarray:
        prompt_len = prompt.shape[1]
        if prompt_len >= self.config.max_seq_len:
            raise ValueError(f"Prompt is too long: {prompt_len} tokens")

        slow_logits, slow_hidden, caches = self._slow_prefill(prompt)

        limit = max_new_tokens or (self.config.max_seq_len - prompt_len)
        limit = min(limit, self.config.max_seq_len - prompt_len)
        history = np.zeros((self.config.num_codebooks + 1, 16), dtype=np.int64)
        frames: list[np.ndarray] = []

        for frame in range(limit):
            semantic = self._sample(
                slow_logits[0, 0], None, temperature, top_p, repetition
            )
            if semantic == self.eos_id:
                break
            semantic_index = max(0, semantic - self.semantic_begin)
            if semantic_index >= self.fast_embedding.shape[0]:
                semantic_index = self.fast_embedding.shape[0] - 1

            fast_caches = self._fast_caches()
            # First fast call warms the fixed-size KV cache; its logits are unused.
            _, fast_caches = self._fast_step(slow_hidden, 0, fast_caches)

            row = [semantic, semantic_index]
            x = self.fast_embedding[semantic_index].reshape(1, 1, -1)
            for codebook in range(1, self.config.num_codebooks):
                logits, fast_caches = self._fast_step(x, codebook, fast_caches)
                previous = torch.from_numpy(history[codebook + 1].copy())
                token = self._sample(
                    logits[0, 0, :1024], previous, temperature, top_p, repetition
                )
                row.append(token)
                x = self.fast_embedding[token].reshape(1, 1, -1)

            frame_tokens = np.asarray(row, dtype=np.int64)
            frames.append(frame_tokens[1:])

            if frame < 16:
                history[:, frame] = frame_tokens
            else:
                history = np.roll(history, -1, axis=1)
                history[:, -1] = frame_tokens

            pos = prompt_len + frame
            if pos >= self.config.max_seq_len:
                break
            tokens = frame_tokens.reshape(1, self.config.num_codebooks + 1, 1)
            slow_logits, slow_hidden, caches = self._slow_step(
                tokens,
                pos=pos,
                caches=caches,
                vq_mask=1.0,
                mask_length=prompt_len + frame + 1,
                allowed_until=prompt_len + frame,
            )

        if not frames:
            return np.zeros((self.config.num_codebooks, 0), dtype=np.int64)
        return np.stack(frames, axis=1)
