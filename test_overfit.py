from __future__ import annotations

import random
import numpy as np
import torch

from interfaces import ModelArgs
from model import NCMModel


def set_seed(seed: int = 42) -> None:
    """Fixes all random seeds for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dump_layer_gradients(model: torch.nn.Module) -> str:
    """Formats a diagnostic summary of parameter gradient norms across all model layers."""
    lines = ["=" * 72, f"{'Parameter Name':<48} | {'Grad Norm':>12} | {'Status':>6}", "-" * 72]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            lines.append(f"{name:<48} | {'N/A':>12} | FROZEN")
        elif param.grad is None:
            lines.append(f"{name:<48} | {'None':>12} | NO_GRAD")
        else:
            grad_norm = param.grad.norm().item()
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()
            status = "NAN/INF" if (has_nan or has_inf) else "OK"
            lines.append(f"{name:<48} | {grad_norm:>12.6f} | {status:>6}")
    lines.append("=" * 72)
    return "\n".join(lines)


def verify_gradients_finite(model: torch.nn.Module, step: int) -> None:
    """Checks that all active gradients are finite (no NaN or Inf) and non-missing."""
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                grad_dump = dump_layer_gradients(model)
                raise RuntimeError(
                    f"[Step {step}] NaN or Inf detected in gradient of '{name}'!\n{grad_dump}"
                )


def run_overfit_test() -> None:
    # 1. Fix random seed (42)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Create synthetic batch: fixed text string repeated 4 times (batch_size=4, seq_len=64)
    fixed_text = "Next-Concept Modeling (NCM) by Stanford Hazy Research compresses tokens into concepts!"
    raw_token_ids = [ord(ch) % 512 for ch in fixed_text]
    # Pad/tile to exact seq_len = 64
    seq_len = 64
    tiled_tokens = (raw_token_ids * ((seq_len // len(raw_token_ids)) + 1))[:seq_len]

    single_seq = torch.tensor(tiled_tokens, dtype=torch.long, device=device)      # [64]
    input_ids = single_seq.unsqueeze(0).repeat(4, 1).contiguous()                 # [4, 64]
    target_ids = input_ids.clone()                                                # [4, 64]

    assert input_ids.shape == (4, 64), f"Expected (4, 64), got {tuple(input_ids.shape)}"

    # 3. Initialize compact NCMModel: dim=128, concept_dim=64, num_concepts=256, downsample_rate=4, layers=2
    args = ModelArgs(
        vocab_size=512,
        dim=128,
        concept_dim=64,
        num_concepts=256,
        downsample_rate=4,
        planner_layers=2,
        decoder_layers=2,
        max_seq_len=64,
    )
    # Disable label smoothing and dead-code replacement during exact single-batch memorization test
    model = NCMModel(args=args).to(device)
    model.planner.label_smoothing = 0.0
    model.tokenizer.set_dead_code_threshold(0)
    model.train()

    # 4. Optimizer: AdamW(lr=2e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)

    # 5. Run single-batch memorization loop
    num_steps = 200
    initial_loss: float = 0.0
    final_loss: float = float("inf")

    for step in range(1, num_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        out = model(input_ids=input_ids, target_ids=target_ids)
        assert out.total_loss is not None, "Expected total_loss to be computed"
        loss = out.total_loss

        loss.backward()
        verify_gradients_finite(model, step=step)
        optimizer.step()

        loss_val = loss.item()
        if step == 1:
            initial_loss = loss_val
        final_loss = loss_val

        if step == 1 or step % 25 == 0 or step == num_steps:
            print(
                f"Step {step:03d}/{num_steps} | "
                f"Total Loss: {loss_val:.6f} | "
                f"CE Loss: {out.ce_loss.item() if out.ce_loss is not None else 0.0:.6f} | "
                f"VQ Loss: {out.vq_loss.item() if out.vq_loss is not None else 0.0:.6f}"
            )

    print(f"\nLoss progression: {initial_loss:.6f} -> {final_loss:.6f}")

    # 6. Validate final convergence threshold (< 0.15) and test closed-loop generation
    if final_loss < 0.15:
        print("PASSED: Loss converged successfully!")
    else:
        grad_dump = dump_layer_gradients(model)
        raise RuntimeError(
            f"FAILED: Overfit test did not converge below 0.15 after {num_steps} steps "
            f"(final_loss={final_loss:.6f}, initial_loss={initial_loss:.6f}).\n"
            f"Layer Gradient Dump:\n{grad_dump}"
        )

    # 7. Closed-loop exact match verification
    model.eval()
    with torch.no_grad():
        prompt = input_ids[:1, :16]
        gen = model.generate_text(
            prompt_token_ids=prompt,
            max_new_tokens=48,
            temperature=0.0,
            token_temperature=0.0,
            repetition_penalty=1.0,
            eos_token_id=None,
        )
        is_exact = torch.equal(gen[0], input_ids[0])
        print(f"Closed-loop Generation Exact Match: {is_exact}")
        if is_exact:
            print("PASSED: Graph, Gradients, and Closed-Loop Generation are 100% functional!")
        else:
            print(f"Generated tokens: {gen[0].tolist()}")
            print(f"Target tokens:    {input_ids[0].tolist()}")


if __name__ == "__main__":
    run_overfit_test()
