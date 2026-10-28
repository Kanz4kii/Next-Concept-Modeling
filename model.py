from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import SemiAutoregressiveDecoder as SurfaceDecoder
from interfaces import ModelArgs, NCMOutput
from planner import ConceptPlanner
from vq_tokenizer import VQConceptTokenizer as ConceptTokenizer


class NCMModel(nn.Module):
    """End-to-End Next-Concept Modeling (NCM v2) with ResidualVQ, RoPE Concept Planner, and Semi-Autoregressive Realizer."""

    def __init__(
        self,
        args: ModelArgs,
        concept_loss_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.dim = args.dim
        self.downsample_rate = args.downsample_rate
        self.concept_loss_weight = concept_loss_weight

        self.text_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.tokenizer = ConceptTokenizer(args=args)
        self.planner = ConceptPlanner(args=args)

        # Select surface realizer based on config (v2 Semi-Autoregressive vs v1 Non-Autoregressive)
        is_v1 = (getattr(args, "decoder_type", "semi_autoregressive") == "non_autoregressive") or (getattr(args, "version", "v2") == "v1")
        if is_v1:
            from decoder import NonAutoregressiveDecoder
            self.decoder = NonAutoregressiveDecoder(args=args)
        else:
            from decoder import SemiAutoregressiveDecoder
            self.decoder = SemiAutoregressiveDecoder(args=args)

        nn.init.normal_(self.text_embeddings.weight, mean=0.0, std=0.02)
        self.tie_weights()

        self.last_active_codes: int = 0
        self.last_codebook_perplexity: float = 0.0
        self._last_batch_counts: Optional[torch.Tensor] = None
        self.window_code_counts: torch.Tensor
        self.register_buffer(
            "window_code_counts",
            torch.zeros(args.num_concepts, dtype=torch.float32),
            persistent=False,
        )

    def compile_submodules(self, mode: str = "default") -> None:
        """Compiles the pure-GPU Encoder Conv stack, ConceptPlanner, and SurfaceDecoder via TorchInductor Triton."""
        self.tokenizer._encode_conv = torch.compile(self.tokenizer._encode_conv, mode=mode)  # type: ignore[method-assign]
        self.planner = torch.compile(self.planner, mode=mode)  # type: ignore[assignment]
        self.decoder = torch.compile(self.decoder, mode=mode)  # type: ignore[assignment]

    @torch.no_grad()
    def get_codebook_stats(self) -> tuple[int, int, float]:
        """Synchronizes codebook telemetry only at logging intervals (avoiding per-microbatch CPU-GPU sync)."""
        active_codes = (
            int((self._last_batch_counts > 0).sum().item())
            if self._last_batch_counts is not None
            else self.last_active_codes
        )
        window_active_codes = int((self.window_code_counts > 1e-3).sum().item())
        probs = self.window_code_counts / self.window_code_counts.sum().clamp(min=1.0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        codebook_ppl = float(torch.exp(entropy).item())
        self.last_active_codes = active_codes
        self.last_codebook_perplexity = codebook_ppl
        return active_codes, window_active_codes, codebook_ppl

    def tie_weights(self) -> None:
        """Ties input token embeddings with surface decoder output vocabulary head and planner RVQ heads."""
        decoder_mod = getattr(self.decoder, "_orig_mod", self.decoder)
        planner_mod = getattr(self.planner, "_orig_mod", self.planner)
        decoder_mod.vocab_head.weight = self.text_embeddings.weight
        for emb, head in zip(planner_mod.concept_embs, planner_mod.lm_heads):
            head.weight = emb.weight

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Returns the exact number of unique parameters (accounting for tied weights)."""
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if trainable_only and not p.requires_grad:
                continue
            ptr = p.data_ptr()
            if ptr not in seen:
                seen.add(ptr)
                total += p.numel()
        return total

    @classmethod
    def from_safetensors(
        cls,
        checkpoint_path: Union[str, Path],
        args: Optional[ModelArgs] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> NCMModel:
        """Loads a pretrained NCMModel from a `.safetensors` file, auto-detecting config if omitted."""
        from safetensors.torch import load_file

        ckpt_path = Path(checkpoint_path)
        state_dict = load_file(str(ckpt_path), device=str(device))

        if args is None:
            # 1. Try config.json in the same directory as the checkpoint
            local_config = ckpt_path.parent / "config.json"
            if local_config.exists():
                args = ModelArgs.from_json(local_config)
            else:
                # 2. Automatically infer ModelArgs architecture from state_dict keys and tensor shapes
                has_token_gate = "decoder.token_gate" in state_dict or "decoder.bos_emb" in state_dict
                version = "v2" if has_token_gate else "v1"
                decoder_type = "semi_autoregressive" if has_token_gate else "non_autoregressive"

                vocab_size = state_dict["text_embeddings.weight"].shape[0] if "text_embeddings.weight" in state_dict else 50304
                dim = state_dict["text_embeddings.weight"].shape[1] if "text_embeddings.weight" in state_dict else 768
                concept_dim = state_dict["tokenizer.to_concept.weight"].shape[0] if "tokenizer.to_concept.weight" in state_dict else 384
                num_quantizers = len([k for k in state_dict if "planner.concept_embs." in k and ".weight" in k]) or 4

                p_layers = [int(k.split(".")[2]) for k in state_dict if k.startswith("planner.layers.") and ".attn." in k]
                planner_layers = (max(p_layers) + 1) if p_layers else 5

                d_layers = [int(k.split(".")[2]) for k in state_dict if k.startswith("decoder.layers.") and ".attn." in k]
                decoder_layers = (max(d_layers) + 1) if d_layers else 5

                args = ModelArgs(
                    vocab_size=vocab_size,
                    dim=dim,
                    concept_dim=concept_dim,
                    num_concepts=1024,
                    num_quantizers=num_quantizers,
                    downsample_rate=4,
                    planner_layers=planner_layers,
                    decoder_layers=decoder_layers,
                    num_heads=8,
                    max_seq_len=1024,
                    version=version,
                    decoder_type=decoder_type,
                )

        model = cls(args=args).to(device)
        model.load_state_dict(state_dict, strict=False)
        model.tie_weights()
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> NCMOutput:
        """Full training/evaluation forward pass of the NCM v2 model.

        Args:
            input_ids: Input token IDs of shape [B, T].
            target_ids: Optional target token IDs of shape [B, T] for surface realization loss.
            return_logits: If False (e.g. during training), frees the [B, T, vocab_size] logits tensor
                immediately after computing cross-entropy loss to save ~1.6 GB of VRAM.

        Returns:
            NCMOutput with token logits [B, T, vocab_size] (or lightweight placeholder) and decomposed losses.
        """
        assert isinstance(input_ids, torch.Tensor), f"Expected torch.Tensor for input_ids, got {type(input_ids)}"
        assert input_ids.ndim == 2, f"Expected 2D tensor [B, T] for input_ids, got {input_ids.ndim}D"
        B, T = input_ids.shape
        assert T % self.downsample_rate == 0, (
            f"Sequence length T ({T}) must be divisible by downsample_rate ({self.downsample_rate})"
        )

        # Step 1: Obtain continuous token embeddings [B, T, dim]
        x_emb = self.text_embeddings(input_ids)                                 # [B, T, dim]

        # Step 2: Compress and quantize into discrete concepts via ResidualVQ Tokenizer
        tok_out = self.tokenizer.encode(x_emb)                                  # quantized: [B, K, concept_dim], indices: [B, K, Q] or [B, K]
        quantized = tok_out.quantized                                           # [B, K, concept_dim]
        indices = tok_out.indices                                               # [B, K, Q] or [B, K]
        vq_loss = tok_out.vq_loss                                               # [] (scalar)

        # Track codebook utilization & perplexity asynchronously on GPU (zero CPU-GPU sync per micro-batch)
        with torch.no_grad():
            flat_idx = indices.reshape(-1)
            counts = torch.bincount(flat_idx, minlength=self.args.num_concepts).float()
            self._last_batch_counts = counts
            self.window_code_counts.mul_(0.9).add_(counts)

        # Step 3: Autoregressive Next-Concept Planning (shift concept targets by 1 step)
        K = indices.size(1)
        if K > 1:
            planner_inputs = indices[:, :-1]                                    # [B, K - 1, Q] or [B, K - 1]
            planner_latents = quantized[:, :-1, :]                              # [B, K - 1, concept_dim]
            planner_targets = indices[:, 1:]                                    # [B, K - 1, Q] or [B, K - 1]
            if target_ids is not None:
                block_valid = (target_ids.view(B, K, self.downsample_rate) != -100).any(dim=-1)  # [B, K]
                target_block_valid = block_valid[:, 1:]                         # [B, K - 1]
                if planner_targets.ndim == 3:
                    planner_targets = planner_targets.masked_fill(~target_block_valid.unsqueeze(-1), -100)
                else:
                    planner_targets = planner_targets.masked_fill(~target_block_valid, -100)
            planner_logits, ce_concept_loss = self.planner(
                planner_inputs,
                targets=planner_targets,
                concept_latents=planner_latents,
            )                                                                   # planner_logits: [B, K - 1, ...], ce_concept_loss: []

            # Construct Planner-predicted concept sequence for Scheduled Conditioning (v2)
            with torch.no_grad():
                pred_indices = planner_logits.argmax(dim=-1)                    # [B, K - 1, Q] or [B, K - 1]
            pred_quantized = self.tokenizer.decode(pred_indices).to(dtype=quantized.dtype)
            planner_concept_seq = torch.cat([quantized[:, :1, :], pred_quantized], dim=1)  # [B, K, concept_dim]
        else:
            _, _ = self.planner(indices, concept_latents=quantized)
            ce_concept_loss = torch.zeros_like(vq_loss)                         # []
            planner_concept_seq = quantized

        # Scheduled Concept Conditioning:
        # During training: 45% oracle encoder concepts (for VQ encoder gradient flow),
        #                  45% planner-predicted concepts (eliminates train-inference exposure bias),
        #                  10% concept dropout (ensures fluent causal token transitions).
        if self.training and K > 1:
            rand_u = torch.rand((B, K, 1), device=quantized.device)
            use_planner_mask = (rand_u >= 0.45) & (rand_u < 0.90)
            drop_concept_mask = rand_u >= 0.90
            decoder_concepts = torch.where(use_planner_mask, planner_concept_seq, quantized)
            decoder_concepts = torch.where(drop_concept_mask, torch.zeros_like(decoder_concepts), decoder_concepts)
        else:
            # During evaluation, use planner_concept_seq so Val PPL reflects true generative quality
            decoder_concepts = planner_concept_seq

        # Step 4: Semi-Autoregressive Surface Realization -> [B, T, vocab_size]
        prev_token_emb = x_emb[:, :-1, :]                                       # [B, T - 1, dim]
        logits = self.decoder(
            decoder_concepts,
            seq_len=T,
            prev_token_emb=prev_token_emb,
        )                                                                       # [B, T, vocab_size]

        # Step 5: Compute losses if target_ids are provided
        total_loss: Optional[torch.Tensor] = None
        ce_loss: Optional[torch.Tensor] = None
        ce_text_loss: Optional[torch.Tensor] = None

        if target_ids is not None:
            assert target_ids.shape == (B, T), (
                f"Expected target_ids shape {(B, T)}, got {tuple(target_ids.shape)}"
            )
            # Compute cross-entropy directly on native bfloat16 logits without redundant FP32 upcast (saves 1.6 GB VRAM)
            with torch.amp.autocast("cuda", enabled=False):
                ce_text_loss = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),                        # [B * T, vocab_size]
                    target_ids.reshape(-1).long(),                              # [B * T]
                ).float()                                                       # []
            if not return_logits:
                logits = logits[:, :1, :1].detach()
            assert ce_concept_loss is not None
            ce_loss = ce_text_loss + self.concept_loss_weight * ce_concept_loss # []
            total_loss = ce_loss + vq_loss                                      # []

        output = NCMOutput(
            logits=logits,                                                      # [B, T, vocab_size] or [B, 1, 1]
            total_loss=total_loss,                                              # [] or None
            vq_loss=vq_loss,                                                    # []
            ce_loss=ce_loss,                                                    # [] or None
            text_loss=ce_text_loss,                                             # [] or None
            concept_loss=ce_concept_loss,                                       # [] or None
        )
        return output.validate()

    @torch.compiler.disable
    @torch.no_grad()
    def generate_text(
        self,
        prompt_token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        min_p: Optional[float] = 0.05,
        token_temperature: float = 0.7,
        repetition_penalty: float = 1.25,
        repetition_window: int = 64,
        eos_token_id: Optional[int] = 50256,
    ) -> torch.Tensor:
        """Closed-Loop Block-Autoregressive Generation (NCM v2).

        Interleaves high-level Concept Planning (1 concept step per `r` tokens) with
        KV-cached Semi-Autoregressive Token Realization (`r` causal token steps per concept),
        re-grounding the concept sequence in the actual generated tokens after each block.
        """
        assert isinstance(prompt_token_ids, torch.Tensor), "prompt_token_ids must be a torch.Tensor"
        assert prompt_token_ids.ndim == 2, f"Expected 2D tensor [B, T_prompt], got {prompt_token_ids.ndim}D"
        assert max_new_tokens > 0, "max_new_tokens must be positive"

        was_training = self.training
        self.eval()

        B, T_prompt = prompt_token_ids.shape
        r = self.downsample_rate

        # Align prompt on the left so its length is a multiple of downsample_rate (r)
        remainder = T_prompt % r
        if remainder != 0:
            pad_len = r - remainder
            pad_tokens = prompt_token_ids[:, :1].expand(B, pad_len)             # [B, pad_len]
            aligned_tokens = torch.cat([pad_tokens, prompt_token_ids], dim=1)   # [B, T_aligned]
        else:
            aligned_tokens = prompt_token_ids.clone()                           # [B, T_prompt]

        # 1. Encode aligned prompt into initial discrete concepts & latents
        prompt_emb = self.text_embeddings(aligned_tokens)                       # [B, T_aligned, dim]
        tok_out = self.tokenizer.encode(prompt_emb)
        current_concepts = tok_out.indices                                      # [B, K_prompt, Q] or [B, K_prompt]
        prompt_latents = tok_out.quantized                                      # [B, K_prompt, concept_dim]

        decoder_mod = getattr(self.decoder, "_orig_mod", self.decoder)
        planner_mod = getattr(self.planner, "_orig_mod", self.planner)

        # 2. Prefill SurfaceDecoder KV-Cache over the aligned prompt
        decoder_kv_caches = decoder_mod.prefill_kv_cache(
            prompt_concept_latents=prompt_latents,
            aligned_prompt_ids=aligned_tokens,
            text_embeddings=self.text_embeddings,
        )

        cur_token_pos = aligned_tokens.size(1)
        last_token_id = aligned_tokens[:, -1:]                                  # [B, 1]
        history_ids = prompt_token_ids.clone()                                  # [B, T_prompt + generated]

        num_new_concepts = math.ceil(max_new_tokens / r)
        tokens_generated = 0

        # 3. Interleaved Block-Autoregressive Loop: 1 Concept Step -> r Token Steps -> Grounding
        for _ in range(num_new_concepts):
            tokens_remaining = max_new_tokens - tokens_generated
            if tokens_remaining <= 0:
                break

            # 3a. Planner predicts the next 1 concept block
            next_concept_seq = planner_mod.generate(
                prompt_indices=current_concepts,
                max_new_concepts=1,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                concept_decoder=self.tokenizer.decode,
            )
            planned_concept = next_concept_seq[:, -1:]                          # [B, 1, Q] or [B, 1]
            block_latent = self.tokenizer.decode(planned_concept)               # [B, 1, concept_dim]

            # 3b. SurfaceDecoder autoregressively realizes up to `r` tokens for this concept block
            block_tokens, last_token_id, decoder_kv_caches = decoder_mod.decode_concept_block(
                block_concept_latent=block_latent,
                last_token_id=last_token_id,
                text_embeddings=self.text_embeddings,
                kv_caches=decoder_kv_caches,
                start_pos=cur_token_pos,
                history_ids=history_ids,
                max_tokens_in_block=tokens_remaining,
                temperature=token_temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                repetition_window=repetition_window,
            )

            cur_token_pos += block_tokens.size(1)
            tokens_generated += block_tokens.size(1)
            history_ids = torch.cat([history_ids, block_tokens], dim=1)
            aligned_tokens = torch.cat([aligned_tokens, block_tokens], dim=1)

            # Check for EOS token if batch size == 1
            if eos_token_id is not None and B == 1:
                eos_positions = (block_tokens[0] == eos_token_id).nonzero(as_tuple=True)[0]
                if eos_positions.numel() > 0:
                    first_eos = int(eos_positions[0].item())
                    # Trim history_ids up to the first EOS token
                    trim_keep = history_ids.size(1) - block_tokens.size(1) + first_eos
                    history_ids = history_ids[:, :trim_keep]
                    break

            # 3c. Re-ground concept sequence with the actual generated tokens if full block completed
            if block_tokens.size(1) == r:
                # Re-encode the last 4 blocks (16 tokens) for exact causal conv context
                tail_tokens = aligned_tokens[:, -min(aligned_tokens.size(1), 4 * r) :]
                tail_emb = self.text_embeddings(tail_tokens)
                grounded_last_concept = self.tokenizer.encode(tail_emb).indices[:, -1:]
                current_concepts = torch.cat([current_concepts, grounded_last_concept], dim=1)
            else:
                current_concepts = next_concept_seq

        if was_training:
            self.train()

        return history_ids                                                      # [B, T_prompt + generated_len]


if __name__ == "__main__":
    torch.manual_seed(42)

    args = ModelArgs(num_quantizers=4)
    model = NCMModel(args=args)

    B, T = 2, 64
    input_ids = torch.randint(0, args.vocab_size, (B, T))                       # [2, 64]
    target_ids = torch.randint(0, args.vocab_size, (B, T))                      # [2, 64]

    out = model(input_ids=input_ids, target_ids=target_ids)
    assert out.logits.shape == (B, T, args.vocab_size), f"Unexpected logits shape: {tuple(out.logits.shape)}"
    assert out.total_loss is not None and out.total_loss.ndim == 0

    generated_ids = model.generate_text(input_ids[:, :16], max_new_tokens=32, temperature=0.8, eos_token_id=None)
    assert generated_ids.shape == (B, 48), f"Unexpected generated_ids shape: {tuple(generated_ids.shape)}"

    print(f"Unique parameters:    {model.num_parameters():,}")
    print(f"Forward logits shape: {tuple(out.logits.shape)}")
    print(f"Total Loss:           {out.total_loss.item():.4f}")
    print(f"Text Loss:            {out.text_loss.item() if out.text_loss is not None else 0.0:.4f}")
    print("End-to-End NCMModel v2 (Semi-Autoregressive) sanity check passed!")
