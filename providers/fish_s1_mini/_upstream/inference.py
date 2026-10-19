"""Dual-AR sampling loop (adapted from fish-speech's inference.py)."""

from collections.abc import Iterator
from typing import Optional

import torch

from .content_sequence import ContentSequence, TextPart, VQPart
from .llama import DualARTransformer
from .text import split_text
from .tokenizer import IM_END_TOKEN


def _multinomial_sample_one_no_sync(probs_sort: torch.Tensor) -> torch.Tensor:
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def _logits_to_probs(
    logits: torch.Tensor,
    previous_tokens: Optional[torch.Tensor],
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
) -> torch.Tensor:
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=0, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=0, index=previous_tokens, src=score)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(
        torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
    )
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=0, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    logits = logits / max(temperature, 1e-5)
    return torch.nn.functional.softmax(logits, dim=-1)


def _sample(
    logits: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> torch.Tensor:
    probs = _logits_to_probs(
        logits=logits[0, -1], previous_tokens=previous_tokens, **sampling_kwargs
    )
    return _multinomial_sample_one_no_sync(probs)


def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> torch.Tensor:
    x = model.forward_generate(x, input_pos)

    codebooks = [
        _sample(
            x.logits,
            # No repetition penalty for the semantic codebook.
            previous_tokens=None,
            **sampling_kwargs,
        )
    ]

    hidden_states = x.hidden_states

    # The fast transformer reuses its KV cache for every frame.
    for layer in model.fast_layers:
        layer.attention.kv_cache.k_cache.fill_(0)
        layer.attention.kv_cache.v_cache.fill_(0)

    input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos)
    semantic = codebooks[0] - model.tokenizer.semantic_begin_id
    semantic[semantic < 0] = 0
    hidden_states = model.fast_embeddings(semantic)
    codebooks.append(semantic)

    for codebook_idx in range(1, model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)
        chunked_logits = logits[..., :1024]
        index = _sample(
            chunked_logits,
            previous_tokens=(
                previous_tokens[codebook_idx + 1]
                if previous_tokens is not None
                else None
            ),
            **sampling_kwargs,
        )
        hidden_states = model.fast_embeddings(index)
        codebooks.append(index)

    return torch.stack(codebooks, dim=0)


def decode_n_tokens(
    model: DualARTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    decode_one_token=decode_one_token_ar,
    **sampling_kwargs,
) -> torch.Tensor:
    previous_tokens = torch.zeros(
        (model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )

    for i in range(num_new_tokens):
        window_size = 16
        if i < window_size:
            window = previous_tokens[:, :window_size]
        else:
            window = previous_tokens[:, i - window_size : i]

        next_token = decode_one_token(
            model=model,
            x=cur_token,
            input_pos=input_pos,
            previous_tokens=window,
            **sampling_kwargs,
        )

        input_pos += 1
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
        previous_tokens[:, i : i + 1] = next_token.view(
            model.config.num_codebooks + 1, -1
        )

        if cur_token[0, 0, -1] == model.tokenizer.get_token_id(IM_END_TOKEN):
            break

    return previous_tokens[:, : i + 1]


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: DualARTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    decode_one_token=decode_one_token_ar,
    **sampling_kwargs,
) -> torch.Tensor:
    """Continue a prompt matrix and return the full token matrix."""
    prompt_length = prompt.size(1)
    if max_new_tokens:
        if prompt_length + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - prompt_length
        if max_new_tokens <= 0:
            raise ValueError("Prompt leaves no room for new tokens")
    else:
        max_new_tokens = model.config.max_seq_len - prompt_length

    device, dtype = prompt.device, prompt.dtype
    codebook_dim = 1 + model.config.num_codebooks

    empty = torch.empty(
        (codebook_dim, model.config.max_seq_len), dtype=dtype, device=device
    )
    empty[:, :prompt_length] = prompt
    sequence = empty
    input_pos = torch.arange(0, prompt_length, device=device)

    next_token = decode_one_token(
        model,
        prompt.view(1, codebook_dim, -1),
        input_pos,
        **sampling_kwargs,
    )
    sequence[:, prompt_length : prompt_length + 1] = next_token

    input_pos = torch.tensor([prompt_length], device=device, dtype=torch.int)
    generated = decode_n_tokens(
        model,
        next_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        decode_one_token=decode_one_token,
        **sampling_kwargs,
    )
    sequence = sequence[:, : prompt_length + 1 + generated.size(1)]
    sequence[:, prompt_length + 1 :] = generated
    return sequence


@torch.no_grad()
@torch.inference_mode()
def generate_segments(
    *,
    model: DualARTransformer,
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
    decode_one_token=decode_one_token_ar,
) -> Iterator[torch.Tensor]:
    """Yield ``(num_codebooks, frames)`` code matrices, one per text segment."""
    if seed is not None:
        torch.manual_seed(seed)

    tokenizer = model.tokenizer
    device = next(model.parameters()).device

    base = ContentSequence(modality="interleave")
    if prompt_text is not None and prompt_tokens is not None:
        base.append([TextPart(text=prompt_text), VQPart(codes=prompt_tokens)], add_end=True)

    encoded_prompt = base.encode_for_inference(
        tokenizer, num_codebooks=model.config.num_codebooks
    )
    if encoded_prompt.size(1) > model.config.max_seq_len - 2048:
        raise ValueError(
            f"Voice prompt is too long: {encoded_prompt.size(1)} tokens"
        )

    texts = split_text(text, chunk_length) if iterative_prompt else [text]
    if not texts:
        raise ValueError("Text contains no speakable content")

    encoded = [
        ContentSequence()
        .append(TextPart(text=segment))
        .encode_for_inference(tokenizer, num_codebooks=model.config.num_codebooks)
        for segment in texts
    ]

    sampling_kwargs = {
        "temperature": torch.tensor(temperature, device=device, dtype=torch.float),
        "top_p": torch.tensor(top_p, device=device, dtype=torch.float),
        "repetition_penalty": torch.tensor(
            repetition_penalty, device=device, dtype=torch.float
        ),
    }

    for segment in encoded:
        prompt = torch.cat([encoded_prompt, segment], dim=1).to(device)
        prompt_length = prompt.size(1)
        output = generate(
            model=model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            decode_one_token=decode_one_token,
            **sampling_kwargs,
        )
        codes = output[1:, prompt_length:-1].clone()
        if codes.size(1):
            yield codes
