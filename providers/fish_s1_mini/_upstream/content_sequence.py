"""Inference-only subset of fish-speech's content sequence encoding."""

from dataclasses import dataclass

import torch

from .tokenizer import IM_END_TOKEN, MODALITY_TOKENS, FishTokenizer


@dataclass
class TextPart:
    text: str
    type: str = "text"


@dataclass
class VQPart:
    codes: torch.Tensor
    type: str = "vq"


class ContentSequence:
    """Builds the interleaved text/VQ token matrix the dual-AR model consumes."""

    def __init__(self, modality: str | None = None):
        self.parts: list[TextPart | VQPart] = []
        if modality is not None:
            self.parts.append(TextPart(text=MODALITY_TOKENS[modality]))

    def append(
        self, part: TextPart | VQPart | list[TextPart | VQPart], add_end: bool = False
    ) -> "ContentSequence":
        if isinstance(part, list):
            self.parts.extend(part)
        else:
            self.parts.append(part)
        if add_end:
            self.parts.append(TextPart(text=IM_END_TOKEN))
        return self

    def encode_for_inference(
        self, tokenizer: FishTokenizer, num_codebooks: int
    ) -> torch.Tensor:
        """Return a ``(num_codebooks + 1, tokens)`` int matrix.

        Row 0 is the text/semantic token stream; rows 1.. are residual
        codebook ids.
        """
        token_parts: list[torch.Tensor] = []
        vq_masks: list[torch.Tensor] = []
        vq_parts: list[torch.Tensor] = []

        for part in self.parts:
            if isinstance(part, VQPart):
                codes = part.codes.clone().to(torch.int)
                tokens = torch.tensor(
                    [
                        tokenizer.semantic_id_to_token_id[int(code)]
                        for code in codes[0]
                    ],
                    dtype=torch.int,
                )
                vq_parts.append(codes)
                vq_masks.append(torch.ones_like(tokens, dtype=torch.bool))
            else:
                tokens = torch.tensor(tokenizer.encode(part.text), dtype=torch.int)
                vq_masks.append(torch.zeros_like(tokens, dtype=torch.bool))
            token_parts.append(tokens)

        tokens = torch.cat(token_parts)
        values = torch.zeros((num_codebooks + 1, len(tokens)), dtype=torch.int)
        values[0] = tokens

        if vq_parts:
            codes = torch.cat(vq_parts, dim=1)
            mask = torch.cat(vq_masks)
            values[0, mask] = codes[0] + tokenizer.semantic_begin_id
            values[1:, mask] = codes

        return values
