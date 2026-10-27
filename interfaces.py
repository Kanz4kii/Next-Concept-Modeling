from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
import torch.nn as nn


@dataclass
class ModelArgs:
    """Configuration arguments for the Next-Concept Modeling (NCM) architecture."""

    vocab_size: int = 32000
    dim: int = 512
    concept_dim: int = 256
    num_concepts: int = 1024
    num_quantizers: int = 4
    downsample_rate: int = 4
    planner_layers: int = 6
    decoder_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.0
    max_seq_len: int = 2048
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        assert isinstance(self.vocab_size, int) and self.vocab_size > 0, "vocab_size must be a positive int"
        assert isinstance(self.dim, int) and self.dim > 0, "dim must be a positive int"
        assert isinstance(self.concept_dim, int) and self.concept_dim > 0, "concept_dim must be a positive int"
        assert isinstance(self.num_concepts, int) and self.num_concepts > 0, "num_concepts must be a positive int"
        assert isinstance(self.num_quantizers, int) and self.num_quantizers > 0, "num_quantizers must be a positive int"
        assert isinstance(self.downsample_rate, int) and self.downsample_rate > 0, "downsample_rate must be a positive int"
        assert isinstance(self.planner_layers, int) and self.planner_layers > 0, "planner_layers must be a positive int"
        assert isinstance(self.decoder_layers, int) and self.decoder_layers > 0, "decoder_layers must be a positive int"
        assert isinstance(self.num_heads, int) and self.num_heads > 0, "num_heads must be a positive int"
        assert self.dim % self.num_heads == 0, f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.concept_dim % self.num_heads == 0, (
            f"concept_dim ({self.concept_dim}) must be divisible by num_heads ({self.num_heads})"
        )
        assert isinstance(self.max_seq_len, int) and self.max_seq_len > 0, "max_seq_len must be a positive int"
        assert self.max_seq_len % self.downsample_rate == 0, (
            f"max_seq_len ({self.max_seq_len}) must be divisible by downsample_rate ({self.downsample_rate})"
        )


class TokenizerOutput(NamedTuple):
    """Structured output of the Concept Tokenizer (supports single VQ [B, K] and ResidualVQ [B, K, Q])."""

    quantized: torch.Tensor  # [B, T // downsample_rate, concept_dim]
    indices: torch.Tensor    # [B, T // downsample_rate] (if Q==1) or [B, T // downsample_rate, Q] (if Q>1)
    vq_loss: torch.Tensor    # scalar []

    def validate(self) -> TokenizerOutput:
        assert isinstance(self.quantized, torch.Tensor), f"Expected torch.Tensor for quantized, got {type(self.quantized)}"
        assert isinstance(self.indices, torch.Tensor), f"Expected torch.Tensor for indices, got {type(self.indices)}"
        assert isinstance(self.vq_loss, torch.Tensor), f"Expected torch.Tensor for vq_loss, got {type(self.vq_loss)}"

        assert self.quantized.ndim == 3, f"Expected 3D tensor [B, K, concept_dim] for quantized, got {self.quantized.ndim}D"
        assert self.indices.ndim in (2, 3), (
            f"Expected 2D [B, K] or 3D [B, K, Q] tensor for indices, got {self.indices.ndim}D"
        )
        assert self.vq_loss.ndim == 0, f"Expected 0D scalar tensor for vq_loss, got {self.vq_loss.ndim}D"
        assert self.quantized.shape[:2] == self.indices.shape[:2], (
            f"Batch and sequence dims must match between quantized {self.quantized.shape[:2]} and indices {self.indices.shape[:2]}"
        )
        return self


class NCMOutput(NamedTuple):
    """Structured output of the Next-Concept Modeling (NCM) model."""

    logits: torch.Tensor                  # [B, T, vocab_size]
    total_loss: Optional[torch.Tensor] = None
    vq_loss: Optional[torch.Tensor] = None
    ce_loss: Optional[torch.Tensor] = None
    text_loss: Optional[torch.Tensor] = None
    concept_loss: Optional[torch.Tensor] = None

    def validate(self) -> NCMOutput:
        assert isinstance(self.logits, torch.Tensor), f"Expected torch.Tensor for logits, got {type(self.logits)}"
        assert self.logits.ndim == 3, f"Expected 3D tensor [B, T, vocab_size] for logits, got {self.logits.ndim}D"

        for name, loss in (
            ("total_loss", self.total_loss),
            ("vq_loss", self.vq_loss),
            ("ce_loss", self.ce_loss),
            ("text_loss", self.text_loss),
            ("concept_loss", self.concept_loss),
        ):
            if loss is not None:
                assert isinstance(loss, torch.Tensor), f"Expected torch.Tensor for {name}, got {type(loss)}"
                assert loss.ndim == 0, f"Expected 0D scalar tensor for {name}, got {loss.ndim}D"
        return self


class BaseConceptTokenizer(nn.Module, ABC):
    """Abstract base class for Concept Tokenizers mapping continuous sequences to discrete latent concepts."""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def validate_encode_input(x: torch.Tensor) -> None:
        """Validates input tensor type and dimensionality for `encode`."""
        assert isinstance(x, torch.Tensor), f"Expected x to be torch.Tensor, got {type(x)}"
        assert x.ndim == 3, f"Expected 3D input tensor [B, T, dim] for encode, got {x.ndim}D (shape: {tuple(x.shape)})"
        assert x.is_floating_point(), f"Expected floating point tensor for x, got {x.dtype}"

    @staticmethod
    def validate_decode_input(indices: torch.Tensor) -> None:
        """Validates input tensor type and dimensionality for `decode`."""
        assert isinstance(indices, torch.Tensor), f"Expected indices to be torch.Tensor, got {type(indices)}"
        assert indices.ndim in (2, 3), (
            f"Expected 2D [B, K] or 3D [B, K, Q] input tensor for decode, got {indices.ndim}D (shape: {tuple(indices.shape)})"
        )
        assert indices.dtype in (torch.int32, torch.int64), (
            f"Expected integer dtype (int32/int64) for indices, got {indices.dtype}"
        )

    @abstractmethod
    def encode(self, x: torch.Tensor) -> TokenizerOutput:
        """Compresses and quantizes token representations into discrete latent concepts."""
        self.validate_encode_input(x)
        raise NotImplementedError

    @abstractmethod
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstructs continuous concept representations from discrete codebook indices."""
        self.validate_decode_input(indices)
        raise NotImplementedError
