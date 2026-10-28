from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from interfaces import ModelArgs


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE) for relative position awareness across arbitrary sequence lengths."""

    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 10000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim ({head_dim}) must be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        self.max_seq_len = max(self.max_seq_len, seq_len)
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)                           # [max_seq_len, head_dim // 2]
        emb = torch.cat((freqs, freqs), dim=-1)                         # [max_seq_len, head_dim]
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: [B, H, L, head_dim]
        seq_len = q.size(2)
        if start_pos + seq_len > self.max_seq_len:
            self._build_cache(start_pos + seq_len)

        cos = self.cos_cached[:, :, start_pos : start_pos + seq_len].to(dtype=q.dtype)  # [1, 1, L, head_dim]
        sin = self.sin_cached[:, :, start_pos : start_pos + seq_len].to(dtype=q.dtype)  # [1, 1, L, head_dim]

        q_embed = (q * cos) + (self._rotate_half(q) * sin)              # [B, H, L, head_dim]
        k_embed = (k * cos) + (self._rotate_half(k) * sin)              # [B, H, L, head_dim]
        return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, multiple_of: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, D]
        gate = F.silu(self.w_gate(x))           # [B, K, hidden_dim]
        up = self.w_up(x)                       # [B, K, hidden_dim]
        out = self.w_down(gate * up)            # [B, K, D]
        return self.dropout(out)                # [B, K, D]


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with RoPE and optional KV-Caching via PyTorch 2.x SDPA."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(head_dim=self.head_dim, max_seq_len=max_seq_len, theta=rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x: [B, K, D]
        B, K, D = x.shape

        qkv = self.qkv_proj(x)                                          # [B, K, 3 * D]
        qkv = qkv.view(B, K, 3, self.num_heads, self.head_dim)          # [B, K, 3, H, head_dim]
        q, k, v = qkv.unbind(dim=2)                                     # each: [B, K, H, head_dim]

        q = q.transpose(1, 2)                                           # [B, H, K, head_dim]
        k = k.transpose(1, 2)                                           # [B, H, K, head_dim]
        v = v.transpose(1, 2)                                           # [B, H, K, head_dim]

        q, k = self.rope(q, k, start_pos=start_pos)                     # [B, H, K, head_dim]

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            if k_prev.size(2) >= start_pos + K:
                k_prev[:, :, start_pos : start_pos + K, :] = k
                v_prev[:, :, start_pos : start_pos + K, :] = v
                k = k_prev[:, :, : start_pos + K, :]
                v = v_prev[:, :, : start_pos + K, :]
                new_kv_cache = (k_prev, v_prev)
            else:
                k = torch.cat([k_prev, k], dim=2)
                v = torch.cat([v_prev, v], dim=2)
                new_kv_cache = (k, v)
        else:
            new_kv_cache = (k, v)
        is_causal = (start_pos == 0) and (K > 1)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )                                                               # [B, H, K, head_dim]

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, K, D)  # [B, K, D]
        out = self.out_proj(attn_out)                                   # [B, K, D]
        return out, new_kv_cache


class PlannerBlock(nn.Module):
    """Pre-RMSNorm Transformer Decoder Block with RoPE and KV-Cache support."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        )
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim=dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x: [B, K, D]
        attn_out, new_kv = self.attn(self.attn_norm(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + attn_out                        # [B, K, D]
        x = x + self.ffn(self.ffn_norm(x))      # [B, K, D]
        return x, new_kv


def sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
) -> torch.Tensor:
    """Samples discrete indices from logits [..., V] using Temperature, Min-P, Top-K, and Nucleus (Top-P) filtering."""
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled_logits = logits.float() / max(temperature, 1e-5)

    # Fast path when Top-K is enabled: torch.topk already returns values sorted descending,
    # allowing Min-P, Top-P, Softmax, and Multinomial to run on [..., K] instead of [..., V].
    if top_k is not None and 0 < top_k < scaled_logits.size(-1):
        top_k_vals, top_k_indices = torch.topk(scaled_logits, top_k, dim=-1)
        probs = F.softmax(top_k_vals, dim=-1)

        if min_p is not None and 0.0 < min_p < 1.0:
            threshold = min_p * probs[..., :1]
            probs = probs.masked_fill(probs < threshold, 0.0)

        if top_p is not None and 0.0 < top_p < 1.0:
            cumulative_probs = torch.cumsum(probs, dim=-1)
            remove_mask = (cumulative_probs - probs) >= top_p
            remove_mask[..., 0] = False
            probs = probs.masked_fill(remove_mask, 0.0)

        sampled_local = torch.multinomial(probs.reshape(-1, top_k), num_samples=1).view(*logits.shape[:-1], 1)
        return torch.gather(top_k_indices, -1, sampled_local)

    # Full-vocab fallback when top_k is disabled
    if min_p is not None and 0.0 < min_p < 1.0:
        probs = F.softmax(scaled_logits, dim=-1)
        max_probs = probs.max(dim=-1, keepdim=True).values
        scaled_logits = scaled_logits.masked_fill(probs < min_p * max_probs, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        scaled_logits = scaled_logits.masked_fill(indices_to_remove, float("-inf"))

    probs = F.softmax(scaled_logits, dim=-1)
    flat_probs = probs.reshape(-1, probs.size(-1))
    sampled = torch.multinomial(flat_probs, num_samples=1)
    return sampled.view(*logits.shape[:-1], 1)


class ConceptPlanner(nn.Module):
    """Autoregressive Transformer Planner operating in single-stage or multi-stage ResidualVQ concept space."""

    def __init__(
        self,
        args: ModelArgs,
        num_heads: Optional[int] = None,
        dropout: Optional[float] = None,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.args = args
        self.concept_dim = args.concept_dim
        self.num_concepts = args.num_concepts
        self.num_quantizers = args.num_quantizers
        self.max_concepts = args.max_seq_len // args.downsample_rate
        self.label_smoothing = label_smoothing

        heads = num_heads if num_heads is not None else args.num_heads
        drop = dropout if dropout is not None else args.dropout

        # Dedicated embedding table and prediction head per RVQ quantizer stage
        self.concept_embs = nn.ModuleList(
            [nn.Embedding(self.num_concepts, self.concept_dim) for _ in range(self.num_quantizers)]
        )

        self.layers = nn.ModuleList(
            [
                PlannerBlock(
                    dim=self.concept_dim,
                    num_heads=heads,
                    dropout=drop,
                    max_seq_len=self.max_concepts,
                    rope_theta=args.rope_theta,
                )
                for _ in range(args.planner_layers)
            ]
        )
        self.norm = RMSNorm(self.concept_dim)
        self.lm_heads = nn.ModuleList(
            [nn.Linear(self.concept_dim, self.num_concepts, bias=False) for _ in range(self.num_quantizers)]
        )

        # Tie per-quantizer input embeddings and output projection heads
        for emb, head in zip(self.concept_embs, self.lm_heads):
            head.weight = emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Sums concept embeddings across all Q quantizer stages -> [B, K, concept_dim]."""
        if indices.ndim == 2:
            return self.concept_embs[0](indices)                        # [B, K, concept_dim]
        assert indices.ndim == 3 and indices.size(-1) == self.num_quantizers, (
            f"Expected indices [B, K, {self.num_quantizers}], got {tuple(indices.shape)}"
        )
        x = torch.zeros(
            indices.size(0),
            indices.size(1),
            self.concept_dim,
            device=indices.device,
            dtype=self.concept_embs[0].weight.dtype,
        )
        for q_idx, emb in enumerate(self.concept_embs):
            x = x + emb(indices[:, :, q_idx])                           # [B, K, concept_dim]
        return x

    def forward(
        self,
        indices: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        concept_latents: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the Concept Planner.

        Args:
            indices: Discrete concept indices of shape [B, K] (when Q=1) or [B, K, Q] (when Q>1).
            targets: Optional target concept indices of shape matching `indices`.
            concept_latents: Optional quantized concept vectors [B, K, concept_dim].

        Returns:
            Tuple of:
              - logits: [B, K, num_concepts] (if 2D input) or [B, K, Q, num_concepts] (if 3D input).
              - loss: scalar CrossEntropyLoss averaged across quantizer stages, or None.
        """
        if not torch.compiler.is_compiling():
            assert isinstance(indices, torch.Tensor), f"Expected torch.Tensor for indices, got {type(indices)}"
            assert indices.ndim in (2, 3), f"Expected 2D [B, K] or 3D [B, K, Q] tensor, got {indices.ndim}D"
            assert indices.dtype in (torch.int32, torch.int64), f"Expected integer dtype for indices, got {indices.dtype}"

        B, K = indices.shape[:2]
        if not torch.compiler.is_compiling():
            assert K <= self.max_concepts, (
                f"Concept sequence length K ({K}) exceeds max_concepts ({self.max_concepts})"
            )

        x = self._embed_indices(indices)                                # [B, K, concept_dim]
        if concept_latents is not None:
            x = x + concept_latents.detach()                            # [B, K, concept_dim]

        for layer in self.layers:
            x, _ = layer(x)                                             # [B, K, concept_dim]

        x = self.norm(x)                                                # [B, K, concept_dim]

        if indices.ndim == 2:
            logits = self.lm_heads[0](x)                                # [B, K, num_concepts]
            loss: Optional[torch.Tensor] = None
            if targets is not None:
                if not torch.compiler.is_compiling():
                    assert targets.shape == (B, K), f"Expected targets shape {(B, K)}, got {tuple(targets.shape)}"
                with torch.amp.autocast("cuda", enabled=False):
                    loss = F.cross_entropy(
                        logits.reshape(-1, self.num_concepts),
                        targets.reshape(-1).long(),
                    ).float()
            return logits, loss

        # Multi-Quantizer ResidualVQ path: [B, K, Q, num_concepts]
        stage_logits = [head(x) for head in self.lm_heads]              # Q x [B, K, num_concepts]
        logits = torch.stack(stage_logits, dim=2)                       # [B, K, Q, num_concepts]

        loss = None
        if targets is not None:
            if not torch.compiler.is_compiling():
                assert targets.shape == (B, K, self.num_quantizers), (
                    f"Expected targets shape {(B, K, self.num_quantizers)}, got {tuple(targets.shape)}"
                )
            with torch.amp.autocast("cuda", enabled=False):
                loss = F.cross_entropy(
                    logits.reshape(-1, self.num_concepts),              # [B * K * Q, num_concepts]
                    targets.reshape(-1).long(),                         # [B * K * Q]
                ).float()                                               # [] (scalar)

        return logits, loss

    @torch.compiler.disable
    @torch.no_grad()
    def generate(
        self,
        prompt_indices: torch.Tensor,
        max_new_concepts: int,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        min_p: Optional[float] = 0.05,
        use_kv_cache: bool = True,
        concept_decoder: Optional[object] = None,
    ) -> torch.Tensor:
        """Autoregressively generates new discrete concept indices with KV-Caching and Top-K/Top-P sampling.

        Args:
            prompt_indices: Initial concept indices of shape [B, K_prompt] or [B, K_prompt, Q].
            max_new_concepts: Number of new concept steps to generate.
            temperature: Sampling temperature (0.0 for greedy argmax decoding).
            top_k: Top-K filtering threshold.
            top_p: Nucleus (Top-P) cumulative probability threshold.
            use_kv_cache: Whether to use incremental O(1) KV-cache decoding.
            concept_decoder: Optional callable (`tokenizer.decode`) mapping concept indices to quantized latents.

        Returns:
            Generated concept sequence of shape [B, K_prompt + max_new_concepts] (or [..., Q]).
        """
        assert isinstance(prompt_indices, torch.Tensor), "prompt_indices must be a torch.Tensor"
        assert prompt_indices.ndim in (2, 3), f"Expected 2D or 3D prompt_indices, got {prompt_indices.ndim}D"
        assert max_new_concepts >= 0, "max_new_concepts must be non-negative"
        assert temperature >= 0.0, "temperature must be non-negative"

        if max_new_concepts == 0:
            return prompt_indices.clone()

        was_training = self.training
        self.eval()

        is_2d = prompt_indices.ndim == 2
        generated = prompt_indices.clone()
        kv_caches: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = [None] * len(self.layers)
        start_pos = 0

        for step_idx in range(max_new_concepts):
            if not use_kv_cache or generated.size(1) > self.max_concepts:
                context = generated[:, -self.max_concepts :]
                ctx_latents = concept_decoder(context) if callable(concept_decoder) else None
                logits, _ = self.forward(context, concept_latents=ctx_latents)
                next_logits = logits[:, -1]                             # [B, V] or [B, Q, V]
            else:
                cur_tokens = generated if step_idx == 0 else generated[:, -1:]
                x = self._embed_indices(cur_tokens)                     # [B, L_step, concept_dim]
                if callable(concept_decoder):
                    x = x + concept_decoder(cur_tokens).to(dtype=x.dtype)
                for i, layer in enumerate(self.layers):
                    x, kv_caches[i] = layer(x, kv_cache=kv_caches[i], start_pos=start_pos)
                start_pos += cur_tokens.size(1)
                x_last = self.norm(x[:, -1:, :])                        # [B, 1, concept_dim]
                if is_2d:
                    next_logits = self.lm_heads[0](x_last).squeeze(1)   # [B, V]
                else:
                    next_logits = torch.stack(
                        [head(x_last).squeeze(1) for head in self.lm_heads], dim=1
                    )                                                   # [B, Q, V]

            sampled = sample_logits(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
            )                                                           # [B, 1] or [B, Q, 1]
            if not is_2d:
                sampled = sampled.squeeze(-1).unsqueeze(1)              # [B, 1, Q]

            generated = torch.cat([generated, sampled], dim=1)

        if was_training:
            self.train()

        return generated

    @torch.compiler.disable
    @torch.no_grad()
    def prefill_kv_cache(
        self,
        prompt_indices: torch.Tensor,
        prompt_latents: Optional[torch.Tensor] = None,
    ) -> list[Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Populates a static pre-allocated Planner KV-cache over prefix concepts."""
        B = prompt_indices.size(0)
        H = self.layers[0].attn.num_heads
        head_dim = self.layers[0].attn.head_dim
        dtype = torch.bfloat16 if (prompt_indices.is_cuda and torch.is_autocast_enabled()) else self.concept_embs[0].weight.dtype
        kv_caches: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = [
            (
                torch.empty((B, H, self.max_concepts, head_dim), dtype=dtype, device=prompt_indices.device),
                torch.empty((B, H, self.max_concepts, head_dim), dtype=dtype, device=prompt_indices.device),
            )
            for _ in range(len(self.layers))
        ]
        if prompt_indices.size(1) == 0:
            return kv_caches
        x = self._embed_indices(prompt_indices)
        if prompt_latents is not None:
            x = x + prompt_latents.to(dtype=x.dtype)
        for i, layer in enumerate(self.layers):
            x, kv_caches[i] = layer(x, kv_cache=kv_caches[i], start_pos=0)
        return kv_caches

    @torch.compiler.disable
    @torch.no_grad()
    def step_kv_cache(
        self,
        cur_concept_idx: torch.Tensor,
        cur_concept_latent: Optional[torch.Tensor],
        kv_caches: list[Optional[tuple[torch.Tensor, torch.Tensor]]],
        start_pos: int,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        min_p: Optional[float] = 0.05,
    ) -> tuple[torch.Tensor, list[Optional[tuple[torch.Tensor, torch.Tensor]]]]:
        """Executes a single O(1) KV-cached concept prediction step."""
        is_2d = cur_concept_idx.ndim == 2
        x = self._embed_indices(cur_concept_idx)                        # [B, 1, concept_dim]
        if cur_concept_latent is not None:
            x = x + cur_concept_latent.to(dtype=x.dtype)
        for i, layer in enumerate(self.layers):
            x, kv_caches[i] = layer(x, kv_cache=kv_caches[i], start_pos=start_pos)
        x_last = self.norm(x[:, -1:, :])                                # [B, 1, concept_dim]
        if is_2d:
            next_logits = self.lm_heads[0](x_last).squeeze(1)           # [B, V]
        else:
            next_logits = torch.stack(
                [head(x_last).squeeze(1) for head in self.lm_heads], dim=1
            )                                                           # [B, Q, V]

        sampled = sample_logits(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
        )
        if not is_2d:
            sampled = sampled.squeeze(-1).unsqueeze(1)                  # [B, 1, Q]
        return sampled, kv_caches


if __name__ == "__main__":
    torch.manual_seed(42)

    args = ModelArgs(num_quantizers=4)
    planner = ConceptPlanner(args=args)

    B, K, Q = 2, 16, 4
    indices = torch.randint(0, args.num_concepts, (B, K, Q))
    targets = torch.randint(0, args.num_concepts, (B, K, Q))

    logits, loss = planner(indices, targets=targets)
    assert logits.shape == (B, K, Q, args.num_concepts)
    assert loss is not None and loss.ndim == 0

    generated = planner.generate(indices[:, :4], max_new_concepts=12, temperature=0.8)
    assert generated.shape == (B, 16, Q)

    print(f"RVQ Planner Logits shape: {tuple(logits.shape)}")
    print(f"RVQ Planner CE Loss:      {loss.item():.4f}")
    print(f"Generated shape (KV):     {tuple(generated.shape)}")
    print("Planner sanity check passed!")
