from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ, VectorQuantize

from interfaces import BaseConceptTokenizer, ModelArgs, TokenizerOutput


class CausalConv1d(nn.Module):
    """1D Causal Convolution ensuring zero information leakage from future tokens/blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.causal_padding = max(0, (kernel_size - 1) * dilation - (stride - 1))
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T]
        x = F.pad(x, (self.causal_padding, 0))  # [B, C_in, T + causal_padding]
        x = self.conv(x)                        # [B, C_out, T // stride]
        return x


class CausalResNetBlock1d(nn.Module):
    """1D Causal ResNet block with GroupNorm and SiLU activation."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        num_groups = 32 if channels % 32 == 0 else min(8, channels)
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=channels),
            nn.SiLU(),
            CausalConv1d(channels, channels, kernel_size=kernel_size, stride=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=channels),
            nn.SiLU(),
            CausalConv1d(channels, channels, kernel_size=1, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        residual = x              # [B, C, T]
        out = self.block(x)       # [B, C, T]
        return residual + out     # [B, C, T]


class VQConceptTokenizer(BaseConceptTokenizer):
    """Residual Vector-Quantized (RVQ) Concept Tokenizer compressing token sequences into multi-stage discrete concepts."""

    def __init__(
        self,
        args: ModelArgs,
        decay: float = 0.99,
        commitment_weight: float = 0.25,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 1,
        quantize_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.concept_dim = args.concept_dim
        self.num_concepts = args.num_concepts
        self.num_quantizers = args.num_quantizers
        self.downsample_rate = args.downsample_rate

        # Encoder: Causal ResNet block + strided CausalConv1d downsampler + projection to concept_dim
        self.encoder_res = CausalResNetBlock1d(channels=self.dim, kernel_size=3)
        self.downsampler = CausalConv1d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=self.downsample_rate * 2,
            stride=self.downsample_rate,
        )
        self.encoder_post_res = CausalResNetBlock1d(channels=self.dim, kernel_size=3)
        self.to_concept = nn.Linear(self.dim, self.concept_dim)
        self.encoder_norm = nn.LayerNorm(self.concept_dim)

        # Multi-Stage ResidualVQ (when num_quantizers > 1) or Single-Stage VectorQuantize (when num_quantizers == 1)
        if self.num_quantizers > 1:
            self.vq: nn.Module = ResidualVQ(
                dim=self.concept_dim,
                num_quantizers=self.num_quantizers,
                codebook_size=self.num_concepts,
                decay=decay,
                commitment_weight=commitment_weight,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                threshold_ema_dead_code=threshold_ema_dead_code,
                use_cosine_sim=True,
                rotation_trick=True,
                quantize_dropout=quantize_dropout,
            )
        else:
            self.vq = VectorQuantize(
                dim=self.concept_dim,
                codebook_size=self.num_concepts,
                decay=decay,
                commitment_weight=commitment_weight,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                threshold_ema_dead_code=threshold_ema_dead_code,
                use_cosine_sim=True,
                rotation_trick=True,
            )

        # Standalone Concept Decoder: projection back to dim + ConvTranspose1d upsampler + ResNet refinement
        self.from_concept = nn.Linear(self.concept_dim, self.dim)
        self.upsampler = nn.ConvTranspose1d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=self.downsample_rate,
            stride=self.downsample_rate,
        )
        self.decoder_res = CausalResNetBlock1d(channels=self.dim, kernel_size=3)
        self.decoder_norm = nn.LayerNorm(self.dim)

    def set_dead_code_threshold(self, threshold: int) -> None:
        """Updates dead code revival threshold across all internal quantizers."""
        if self.num_quantizers > 1:
            for layer in self.vq.layers:
                layer._codebook.threshold_ema_dead_code = threshold
        else:
            self.vq._codebook.threshold_ema_dead_code = threshold

    @torch.compiler.disable
    def _quantize_step(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Executes VQ/RVQ codebook lookup and EMA update outside TorchDynamo to prevent graph breaks."""
        quantized, indices, vq_loss = self.vq(z_e)
        return quantized, indices, vq_loss.mean()

    def _encode_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Pure GPU tensor encoder stack (compilable by TorchInductor Triton without C++ cl.exe)."""
        h = x.transpose(1, 2).contiguous()          # [B, dim, T]
        h = self.encoder_res(h)                     # [B, dim, T]
        h = self.downsampler(h)                     # [B, dim, T // downsample_rate]
        h = self.encoder_post_res(h)                # [B, dim, T // downsample_rate]
        h = h.transpose(1, 2).contiguous()          # [B, T // downsample_rate, dim]
        z_e = self.to_concept(h)                    # [B, T // downsample_rate, concept_dim]
        z_e = self.encoder_norm(z_e)                # [B, T // downsample_rate, concept_dim]
        return z_e

    def encode(self, x: torch.Tensor) -> TokenizerOutput:
        """Encodes continuous token embeddings into discrete concepts.

        Args:
            x: Token embeddings of shape [B, T, dim].

        Returns:
            TokenizerOutput with quantized [B, K, concept_dim], indices [B, K, Q] (or [B, K]), and scalar vq_loss.
        """
        if not torch.compiler.is_compiling():
            self.validate_encode_input(x)  # x: [B, T, dim]
            assert x.size(-1) == self.dim, f"Expected feature dim {self.dim}, got {x.size(-1)}"
            assert x.size(1) % self.downsample_rate == 0, (
                f"Sequence length T ({x.size(1)}) must be divisible by downsample_rate ({self.downsample_rate})"
            )

        z_e = self._encode_conv(x)                  # [B, T // downsample_rate, concept_dim]
        quantized, indices, vq_loss = self._quantize_step(z_e)

        return TokenizerOutput(
            quantized=quantized,                    # [B, T // downsample_rate, concept_dim]
            indices=indices,                        # [B, T // downsample_rate, Q] or [B, T // downsample_rate]
            vq_loss=vq_loss,                        # []
        )

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Lookups quantized concept vectors from discrete codebook indices.

        Args:
            indices: Discrete concept indices of shape [B, K, Q] or [B, K].

        Returns:
            Quantized concept vectors of shape [B, K, concept_dim].
        """
        self.validate_decode_input(indices)
        if self.num_quantizers > 1 and indices.ndim == 2:
            indices = indices.unsqueeze(-1)         # [B, K, 1]
        quantized = self.vq.get_output_from_indices(indices)  # [B, K, concept_dim]
        return quantized                            # [B, K, concept_dim]

    def decode_to_tokens(self, quantized: torch.Tensor) -> torch.Tensor:
        """Upsamples quantized concepts back to original sequence resolution T for standalone tokenizer training.

        Args:
            quantized: Quantized concept representations of shape [B, T // downsample_rate, concept_dim].

        Returns:
            Reconstructed token-level representations of shape [B, T, dim].
        """
        assert isinstance(quantized, torch.Tensor), f"Expected torch.Tensor, got {type(quantized)}"
        assert quantized.ndim == 3, f"Expected 3D tensor [B, K, concept_dim], got {quantized.ndim}D"
        assert quantized.size(-1) == self.concept_dim, (
            f"Expected concept_dim {self.concept_dim}, got {quantized.size(-1)}"
        )

        h = self.from_concept(quantized)            # [B, T // downsample_rate, dim]
        h = h.transpose(1, 2).contiguous()          # [B, dim, T // downsample_rate]
        h = self.upsampler(h)                       # [B, dim, T]
        h = self.decoder_res(h)                     # [B, dim, T]
        h = h.transpose(1, 2).contiguous()          # [B, T, dim]
        out = self.decoder_norm(h)                  # [B, T, dim]
        return out                                  # [B, T, dim]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, TokenizerOutput]:
        """Full autoencoder forward pass for standalone VQ tokenizer pretraining."""
        tok_out = self.encode(x)                            # quantized: [B, K, concept_dim], indices: [B, K, Q]
        x_recon = self.decode_to_tokens(tok_out.quantized)  # [B, T, dim]
        return x_recon, tok_out


if __name__ == "__main__":
    torch.manual_seed(42)

    # 1. Verify Multi-Quantizer ResidualVQ (num_quantizers=4)
    args_rvq = ModelArgs(dim=512, concept_dim=256, num_concepts=1024, num_quantizers=4, downsample_rate=4)
    tokenizer_rvq = VQConceptTokenizer(args=args_rvq)
    x = torch.randn(2, 64, 512)                             # [B=2, T=64, dim=512]
    x_recon, out_rvq = tokenizer_rvq(x)
    assert out_rvq.indices.shape == (2, 16, 4), f"Expected (2, 16, 4), got {tuple(out_rvq.indices.shape)}"
    assert out_rvq.quantized.shape == (2, 16, 256)
    assert tokenizer_rvq.decode(out_rvq.indices).shape == (2, 16, 256)

    # 2. Verify Single-Quantizer compatibility (num_quantizers=1 -> [2, 16])
    args_single = ModelArgs(dim=512, concept_dim=256, num_concepts=1024, num_quantizers=1, downsample_rate=4)
    tokenizer_single = VQConceptTokenizer(args=args_single)
    _, out_single = tokenizer_single(x)
    assert out_single.indices.shape == (2, 16), f"Expected (2, 16), got {tuple(out_single.indices.shape)}"

    print(f"Input shape:            {tuple(x.shape)}")
    print(f"RVQ Indices shape:      {tuple(out_rvq.indices.shape)} (40 bits/concept)")
    print(f"SingleVQ Indices shape: {tuple(out_single.indices.shape)} (10 bits/concept)")
    print(f"Quantized shape:        {tuple(out_rvq.quantized.shape)}")
    print(f"Reconstructed shape:    {tuple(x_recon.shape)}")
    print("All VQ / ResidualVQ assertions passed successfully!")
