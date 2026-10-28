from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from interfaces import ModelArgs
from planner import RMSNorm, RotaryEmbedding, SwiGLU, sample_logits


class SurfaceSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with RoPE and KV-Cache support via PyTorch 2.x SDPA."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        is_causal: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.is_causal = is_causal

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(head_dim=self.head_dim, max_seq_len=max_seq_len, theta=rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x: [B, T, dim]
        B, T, D = x.shape

        qkv = self.qkv_proj(x)                                          # [B, T, 3 * dim]
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)          # [B, T, 3, H, head_dim]
        q, k, v = qkv.unbind(dim=2)                                     # each: [B, T, H, head_dim]

        q = q.transpose(1, 2)                                           # [B, H, T, head_dim]
        k = k.transpose(1, 2)                                           # [B, H, T, head_dim]
        v = v.transpose(1, 2)                                           # [B, H, T, head_dim]

        q, k = self.rope(q, k, start_pos=start_pos)                     # [B, H, T, head_dim]

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            if k_prev.size(2) >= start_pos + T:
                k_prev[:, :, start_pos : start_pos + T, :] = k
                v_prev[:, :, start_pos : start_pos + T, :] = v
                k = k_prev[:, :, : start_pos + T, :]
                v = v_prev[:, :, : start_pos + T, :]
                new_kv_cache = (k_prev, v_prev)
            else:
                k = torch.cat([k_prev, k], dim=2)
                v = torch.cat([v_prev, v], dim=2)
                new_kv_cache = (k, v)
        else:
            new_kv_cache = (k, v)
        use_causal = self.is_causal and (start_pos == 0) and (T > 1)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_causal,
        )                                                               # [B, H, T, head_dim]

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, dim]
        out = self.out_proj(attn_out)                                   # [B, T, dim]
        return out, new_kv_cache


class DecoderBlock(nn.Module):
    """Pre-RMSNorm Transformer Block for Semi-Autoregressive Surface Realization."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        is_causal: bool = True,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = SurfaceSelfAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            is_causal=is_causal,
        )
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim=dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x: [B, T, dim]
        attn_out, new_kv = self.attn(self.attn_norm(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + attn_out                        # [B, T, dim]
        x = x + self.ffn(self.ffn_norm(x))      # [B, T, D]
        return x, new_kv                        # [B, T, dim]


class ConceptUpsampler(nn.Module):
    """Upsamples latent concepts [B, K, concept_dim] to token resolution [B, T, dim]."""

    def __init__(self, concept_dim: int, dim: int, downsample_rate: int, max_seq_len: int) -> None:
        super().__init__()
        self.concept_dim = concept_dim
        self.dim = dim
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len

        self.in_proj = nn.Linear(concept_dim, dim)
        self.deconv = nn.ConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=downsample_rate,
            stride=downsample_rate,
        )
        self.intra_chunk_pos_emb = nn.Embedding(downsample_rate, dim)
        self.norm = RMSNorm(dim)
        offsets = torch.arange(0, max_seq_len, dtype=torch.long) % downsample_rate
        self.register_buffer("chunk_offsets_buf", offsets, persistent=False)

    def forward(self, concept_latents: torch.Tensor, seq_len: int) -> torch.Tensor:
        # concept_latents: [B, K, concept_dim]
        B, K, _ = concept_latents.shape
        if not torch.compiler.is_compiling():
            upsampled_len = K * self.downsample_rate
            assert upsampled_len >= seq_len, (
                f"Upsampled length K * downsample_rate ({upsampled_len}) cannot be smaller than target seq_len ({seq_len})"
            )

        h = self.in_proj(concept_latents)                                                   # [B, K, dim]

        # Residual repeat_interleave path + learned ConvTranspose1d upsampling path
        h_rep = h.repeat_interleave(self.downsample_rate, dim=1)                            # [B, K * r, dim]
        h_conv = self.deconv(h.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()   # [B, K * r, dim]
        x = (h_rep + h_conv)[:, :seq_len, :]                                                # [B, T, dim]

        if seq_len <= self.chunk_offsets_buf.size(0):
            chunk_offsets = self.chunk_offsets_buf[:seq_len]                                # [T]
        else:
            chunk_offsets = torch.arange(0, seq_len, device=concept_latents.device, dtype=torch.long) % self.downsample_rate

        intra_encoding = self.intra_chunk_pos_emb(chunk_offsets)                            # [T, dim]
        x = self.norm(x + intra_encoding.unsqueeze(0))                                      # [B, T, dim]
        return x                                                                            # [B, T, dim]


class SemiAutoregressiveDecoder(nn.Module):
    """Semi-Autoregressive (Concept-Conditioned Causal) Surface Realizer (NCM v2).

    Combines upsampled high-level concept latents [B, K, concept_dim] with causal shifted
    previous-token embeddings [B, T, dim], enabling parallel teacher-forced training and
    KV-cached block-autoregressive surface realization at inference time.
    """

    def __init__(
        self,
        args: ModelArgs,
        num_heads: Optional[int] = None,
        dropout: Optional[float] = None,
        is_causal: bool = True,
    ) -> None:
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.concept_dim = args.concept_dim
        self.vocab_size = args.vocab_size
        self.downsample_rate = args.downsample_rate
        self.max_seq_len = args.max_seq_len

        heads = num_heads if num_heads is not None else args.num_heads
        drop = dropout if dropout is not None else args.dropout

        self.upsampler = ConceptUpsampler(
            concept_dim=self.concept_dim,
            dim=self.dim,
            downsample_rate=self.downsample_rate,
            max_seq_len=self.max_seq_len,
        )

        # Autoregressive previous-token conditioning stream (v2)
        self.bos_emb = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.token_norm = RMSNorm(self.dim)
        self.concept_gate = nn.Parameter(torch.ones(1, 1, self.dim))
        self.token_gate = nn.Parameter(torch.ones(1, 1, self.dim))

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    dim=self.dim,
                    num_heads=heads,
                    dropout=drop,
                    max_seq_len=self.max_seq_len,
                    rope_theta=args.rope_theta,
                    is_causal=is_causal,
                )
                for _ in range(args.decoder_layers)
            ]
        )
        self.norm = RMSNorm(self.dim)
        self.vocab_head = nn.Linear(self.dim, self.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.bos_emb, mean=0.0, std=0.02)

    def forward(
        self,
        concept_latents: torch.Tensor,
        seq_len: int,
        prev_token_emb: Optional[torch.Tensor] = None,
        token_dropout_p: float = 0.15,
    ) -> torch.Tensor:
        """Parallel teacher-forced realization of token logits conditioned on concepts and previous tokens.

        Args:
            concept_latents: Quantized concept vectors of shape [B, K, concept_dim].
            seq_len: Target token sequence length T (typically T = K * downsample_rate).
            prev_token_emb: Embeddings of previous tokens of shape [B, T - 1, dim] (or [B, T, dim]).
                At position t > 0, receives embedding of token t - 1; position 0 receives `self.bos_emb`.
            token_dropout_p: Word-dropout probability on `prev_token_emb` during training so the
                decoder relies on both `concept_latents` and `prev_token_emb`.

        Returns:
            token_logits of shape [B, T, vocab_size].
        """
        B = concept_latents.size(0)
        if not torch.compiler.is_compiling():
            assert isinstance(concept_latents, torch.Tensor), (
                f"Expected torch.Tensor for concept_latents, got {type(concept_latents)}"
            )
            assert concept_latents.ndim == 3, (
                f"Expected 3D tensor [B, K, concept_dim] for concept_latents, got {concept_latents.ndim}D"
            )
            assert concept_latents.size(-1) == self.concept_dim, (
                f"Expected concept_dim {self.concept_dim}, got {concept_latents.size(-1)}"
            )
            assert isinstance(seq_len, int) and seq_len > 0, f"seq_len must be a positive int, got {seq_len}"

        c_up = self.upsampler(concept_latents, seq_len=seq_len)             # [B, T, dim]

        if prev_token_emb is not None:
            bos = self.bos_emb.expand(B, 1, self.dim).to(dtype=prev_token_emb.dtype)
            if prev_token_emb.size(1) == seq_len - 1:
                t_in = torch.cat([bos, prev_token_emb], dim=1)              # [B, T, dim]
            elif prev_token_emb.size(1) == seq_len:
                t_in = prev_token_emb
            else:
                raise ValueError(
                    f"prev_token_emb length ({prev_token_emb.size(1)}) must be seq_len-1 ({seq_len - 1}) or seq_len ({seq_len})"
                )

            if self.training and token_dropout_p > 0.0:
                drop_mask = torch.rand((B, seq_len, 1), device=t_in.device) < token_dropout_p
                t_in = torch.where(drop_mask, bos.expand(B, seq_len, self.dim), t_in)
        else:
            t_in = self.bos_emb.expand(B, seq_len, self.dim).to(dtype=c_up.dtype)

        x = self.concept_gate * c_up + self.token_gate * self.token_norm(t_in)  # [B, T, dim]

        for layer in self.layers:
            x, _ = layer(x)                                                 # [B, T, dim]

        x = self.norm(x)                                                    # [B, T, dim]
        token_logits = self.vocab_head(x)                                   # [B, T, vocab_size]

        if not torch.compiler.is_compiling():
            assert token_logits.size(1) == seq_len, (
                f"Output sequence length ({token_logits.size(1)}) does not match target seq_len ({seq_len})"
            )
        return token_logits                                                 # [B, T, vocab_size]

    @torch.compiler.disable
    @torch.no_grad()
    def prefill_kv_cache(
        self,
        prompt_concept_latents: torch.Tensor,
        aligned_prompt_ids: torch.Tensor,
        text_embeddings: nn.Embedding,
    ) -> list[Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Populates the causal KV-cache over the aligned prompt tokens [0 .. T_aligned - 1]."""
        B, T_aligned = aligned_prompt_ids.shape
        c_up = self.upsampler(prompt_concept_latents, seq_len=T_aligned)    # [B, T_aligned, dim]
        bos = self.bos_emb.expand(B, 1, self.dim).to(dtype=c_up.dtype)
        if T_aligned > 1:
            prev_emb = text_embeddings(aligned_prompt_ids[:, :-1]).to(dtype=c_up.dtype)
            t_in = torch.cat([bos, prev_emb], dim=1)                        # [B, T_aligned, dim]
        else:
            t_in = bos

        x = self.concept_gate * c_up + self.token_gate * self.token_norm(t_in)
        H = self.layers[0].attn.num_heads
        head_dim = self.layers[0].attn.head_dim
        dtype = torch.bfloat16 if (x.is_cuda and torch.is_autocast_enabled()) else x.dtype
        kv_caches: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = [
            (
                torch.empty((B, H, self.max_seq_len, head_dim), dtype=dtype, device=x.device),
                torch.empty((B, H, self.max_seq_len, head_dim), dtype=dtype, device=x.device),
            )
            for _ in range(len(self.layers))
        ]
        for i, layer in enumerate(self.layers):
            x, kv_caches[i] = layer(x, kv_cache=kv_caches[i], start_pos=0)
        return kv_caches

    @torch.compiler.disable
    @torch.no_grad()
    def decode_concept_block(
        self,
        block_concept_latent: torch.Tensor,
        last_token_id: torch.Tensor,
        text_embeddings: nn.Embedding,
        kv_caches: list[Optional[tuple[torch.Tensor, torch.Tensor]]],
        start_pos: int,
        history_ids: torch.Tensor,
        max_tokens_in_block: Optional[int] = None,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        min_p: Optional[float] = 0.05,
        repetition_penalty: float = 1.25,
        repetition_window: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor, list[Optional[tuple[torch.Tensor, torch.Tensor]]]]:
        """Autoregressively decodes up to `r` tokens for a single planned concept block using KV-Cache."""
        B = block_concept_latent.size(0)
        r = self.downsample_rate
        steps = r if max_tokens_in_block is None else min(r, max_tokens_in_block)

        c_block = self.upsampler(block_concept_latent, seq_len=r)           # [B, r, dim]
        cur_last_token = last_token_id                                      # [B, 1]
        cur_history = history_ids                                           # [B, L_hist]
        block_tokens: list[torch.Tensor] = []

        for j in range(steps):
            c_step = c_block[:, j : j + 1, :]                               # [B, 1, dim]
            t_step = text_embeddings(cur_last_token).to(dtype=c_step.dtype) # [B, 1, dim]
            x_step = self.concept_gate * c_step + self.token_gate * self.token_norm(t_step)

            for i, layer in enumerate(self.layers):
                x_step, kv_caches[i] = layer(x_step, kv_cache=kv_caches[i], start_pos=start_pos + j)

            logits_step = self.vocab_head(self.norm(x_step)).squeeze(1).float()  # [B, vocab_size]
            if logits_step.size(-1) > 50257:
                logits_step[:, 50257:] = float("-inf")

            # Apply repetition penalty without GPU-CPU stream synchronization
            if repetition_penalty != 1.0 and cur_history.size(1) > 0:
                window_tokens = cur_history[:, -repetition_window:]             # [B, W]
                score = torch.gather(logits_step, 1, window_tokens)             # [B, W]
                score = torch.where(
                    score > 0,
                    score / repetition_penalty,
                    score * repetition_penalty,
                )
                logits_step = logits_step.scatter(1, window_tokens, score)      # [B, vocab_size]

            next_token = sample_logits(
                logits_step,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
            )                                                               # [B, 1]
            block_tokens.append(next_token)
            cur_last_token = next_token
            cur_history = torch.cat([cur_history, next_token], dim=1)

        block_token_ids = torch.cat(block_tokens, dim=1)                    # [B, steps]
        return block_token_ids, cur_last_token, kv_caches

    @torch.compiler.disable
    @torch.no_grad()
    def decode_planned_latents(
        self,
        planned_concept_latents: torch.Tensor,
        last_token_id: torch.Tensor,
        text_embeddings: nn.Embedding,
        kv_caches: list[Optional[tuple[torch.Tensor, torch.Tensor]]],
        start_pos: int,
        prompt_token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        min_p: Optional[float] = 0.05,
        repetition_penalty: float = 1.25,
        repetition_window: int = 64,
        eos_token_id: Optional[int] = 50256,
    ) -> torch.Tensor:
        """Fast batched-upsampling KV-cached token realization over a sequence of planned concepts."""
        B, K_new, _ = planned_concept_latents.shape
        T_prompt = prompt_token_ids.size(1)
        total_upsampled = K_new * self.downsample_rate

        # Upsample all planned concepts in a single pass and pre-apply concept_gate
        c_all = self.upsampler(planned_concept_latents, seq_len=total_upsampled)[:, :max_new_tokens, :]
        c_all_gated = self.concept_gate * c_all                             # [B, max_new_tokens, dim]

        # Pre-allocate output token buffer to eliminate per-step torch.cat allocations
        history_buf = torch.empty(
            (B, T_prompt + max_new_tokens),
            dtype=prompt_token_ids.dtype,
            device=prompt_token_ids.device,
        )
        history_buf[:, :T_prompt] = prompt_token_ids
        cur_len = T_prompt
        cur_last_token = last_token_id                                      # [B, 1]

        for j in range(max_new_tokens):
            t_step = text_embeddings(cur_last_token).to(dtype=c_all_gated.dtype)
            x_step = c_all_gated[:, j : j + 1, :] + self.token_gate * self.token_norm(t_step)

            for i, layer in enumerate(self.layers):
                x_step, kv_caches[i] = layer(x_step, kv_cache=kv_caches[i], start_pos=start_pos + j)

            logits_step = self.vocab_head(self.norm(x_step)).squeeze(1)[:, :50257]  # [B, 50257] zero-copy slice

            if repetition_penalty != 1.0 and cur_len > 0:
                win_start = max(0, cur_len - repetition_window)
                window_tokens = history_buf[:, win_start:cur_len]           # [B, W]
                score = torch.gather(logits_step, 1, window_tokens)
                score = torch.where(
                    score > 0,
                    score / repetition_penalty,
                    score * repetition_penalty,
                )
                logits_step = logits_step.scatter(1, window_tokens, score)

            if temperature <= 0.0:
                next_token = torch.argmax(logits_step, dim=-1, keepdim=True)
            else:
                next_token = sample_logits(
                    logits_step,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                )

            history_buf[:, cur_len : cur_len + 1] = next_token
            cur_len += 1
            cur_last_token = next_token

            if eos_token_id is not None and B == 1 and int(next_token.item()) == eos_token_id:
                cur_len -= 1
                break

        return history_buf[:, :cur_len]


class NonAutoregressiveDecoder(nn.Module):
    """Original Parallel Non-Autoregressive Surface Realizer (NCM v1).

    Maintained for complete backward compatibility with NCM v1 checkpoints.
    Predicts surface tokens strictly in parallel from upsampled concept latents.
    """

    def __init__(
        self,
        args: ModelArgs,
        num_heads: Optional[int] = None,
        dropout: Optional[float] = None,
        is_causal: bool = True,
    ) -> None:
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.concept_dim = args.concept_dim
        self.vocab_size = args.vocab_size
        self.downsample_rate = args.downsample_rate
        self.max_seq_len = args.max_seq_len

        heads = num_heads if num_heads is not None else args.num_heads
        drop = dropout if dropout is not None else args.dropout

        self.upsampler = ConceptUpsampler(
            concept_dim=self.concept_dim,
            dim=self.dim,
            downsample_rate=self.downsample_rate,
            max_seq_len=self.max_seq_len,
        )

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    dim=self.dim,
                    num_heads=heads,
                    dropout=drop,
                    max_seq_len=self.max_seq_len,
                    rope_theta=args.rope_theta,
                    is_causal=is_causal,
                )
                for _ in range(args.decoder_layers)
            ]
        )
        self.norm = RMSNorm(self.dim)
        self.vocab_head = nn.Linear(self.dim, self.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        concept_latents: torch.Tensor,
        seq_len: int,
        prev_token_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Parallel realization of token logits from latent concept representations."""
        x = self.upsampler(concept_latents, seq_len=seq_len)    # [B, T, dim]
        for layer in self.layers:
            x, _ = layer(x)                                     # [B, T, dim]
        x = self.norm(x)                                        # [B, T, dim]
        token_logits = self.vocab_head(x)                       # [B, T, vocab_size]
        return token_logits


if __name__ == "__main__":
    torch.manual_seed(42)

    args = ModelArgs()
    decoder = SemiAutoregressiveDecoder(args=args)

    B, K = 2, 16
    T = K * args.downsample_rate                                # 64
    concept_latents = torch.randn(B, K, args.concept_dim)       # [2, 16, 256]
    prev_token_emb = torch.randn(B, T - 1, args.dim)            # [2, 63, 512]

    token_logits = decoder(concept_latents, seq_len=T, prev_token_emb=prev_token_emb)
    assert token_logits.shape == (B, T, args.vocab_size), (
        f"Expected shape {(B, T, args.vocab_size)}, got {tuple(token_logits.shape)}"
    )

    print(f"Input concept latents shape: {tuple(concept_latents.shape)}")
    print(f"Output token logits shape:   {tuple(token_logits.shape)}")
    print("SemiAutoregressiveDecoder (v2) sanity check passed!")
