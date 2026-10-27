from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Optional

import bitsandbytes as bnb
import torch
from safetensors.torch import load_file, save_file
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import wandb

from dataset import MemmapDataset, prepare_fineweb_bin
from interfaces import ModelArgs
from model import NCMModel


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Cosine Annealing LR schedule with linear warmup for the first `num_warmup_steps` steps."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step + 1) / float(max(1, num_warmup_steps))
        if current_step >= num_training_steps:
            return min_lr_ratio
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def estimate_flops_per_token(model: NCMModel, seq_len: int) -> float:
    """Approximates training FLOPs per token (6 * N_params + attention quadratic term)."""
    num_params = model.num_parameters(trainable_only=True)
    args = model.args
    k_len = seq_len // args.downsample_rate
    planner_attn_flops = 12.0 * args.planner_layers * args.concept_dim * k_len / args.downsample_rate
    decoder_attn_flops = 12.0 * args.decoder_layers * args.dim * seq_len
    return 6.0 * num_params + planner_attn_flops + decoder_attn_flops


def save_safetensors_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: Path,
    filename: str,
) -> Path:
    """Saves model weights strictly in `.safetensors` format (cloning shared tensors for tied weights)."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / filename

    raw_model = getattr(model, "_orig_mod", model)
    seen_ptrs: set[int] = set()
    state_dict: dict[str, torch.Tensor] = {}
    for k, v in raw_model.state_dict().items():
        clean_k = k.replace("._orig_mod.", ".")
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            state_dict[clean_k] = v.detach().clone().contiguous().cpu()
        else:
            seen_ptrs.add(ptr)
            state_dict[clean_k] = v.detach().contiguous().cpu()
    save_file(state_dict, str(ckpt_path))
    return ckpt_path


def save_training_state(
    model: NCMModel,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    checkpoint_dir: Path,
    step: int,
    best_val_loss: float,
    wandb_run_id: Optional[str],
    tag: Optional[str] = None,
    keep_last_n: int = 2,
) -> Path:
    """Saves both `.safetensors` model weights and companion trainer state for seamless `--resume`."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    name_prefix = f"ncm_{tag}" if tag is not None else f"ncm_step_{step:06d}"

    # 1. Update `ncm_latest.safetensors` in place (zero disk bloat, instant resume)
    latest_weights_path = save_safetensors_checkpoint(model, checkpoint_dir, filename="ncm_latest.safetensors")

    # 2. Save milestone checkpoint every 5000 steps or when explicit tag is given
    if step % 5000 == 0 or tag is not None:
        save_safetensors_checkpoint(model, checkpoint_dir, filename=f"{name_prefix}.safetensors")

    # 3. Save companion trainer state (optimizer 8-bit buffers, LR scheduler, step, WandB run ID)
    trainer_state: dict[str, Any] = {
        "step": step,
        "best_val_loss": best_val_loss,
        "wandb_run_id": wandb_run_id,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "window_code_counts": model.window_code_counts.detach().cpu(),
        "rng_state_cpu": torch.get_rng_state(),
        "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(trainer_state, checkpoint_dir / "trainer_state_latest.pt")
    if step % 5000 == 0:
        torch.save(trainer_state, checkpoint_dir / f"trainer_state_step_{step:06d}.pt")

    # 4. Auto-rotation: remove older intermediate step checkpoints so disk never runs out of space
    step_ckpts = sorted(checkpoint_dir.glob("ncm_step_*.safetensors"))
    milestones = {"005000", "010000", "015000", "020000", "025000", "030000"}
    if len(step_ckpts) > keep_last_n + len(milestones):
        for old in step_ckpts[:-keep_last_n]:
            if not any(m in old.name for m in milestones):
                try:
                    old.unlink()
                    old_pt = checkpoint_dir / f"trainer_state_{old.stem.replace('ncm_', '')}.pt"
                    if old_pt.exists():
                        old_pt.unlink()
                except Exception:
                    pass

    return latest_weights_path


def load_training_state(
    model: NCMModel,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    checkpoint_dir: Path,
    device: torch.device,
    resume_from: Optional[str] = None,
) -> tuple[int, float, Optional[str]]:
    """Restores model weights from `.safetensors` and optimizer/scheduler state from `trainer_state_latest.pt`."""
    if resume_from is not None and resume_from not in ("auto", "latest", "True", "true"):
        weights_file = Path(resume_from)
        state_file = weights_file.parent / "trainer_state_latest.pt"
    else:
        weights_file = checkpoint_dir / "ncm_latest.safetensors"
        state_file = checkpoint_dir / "trainer_state_latest.pt"
        if not weights_file.exists():
            # Fallback to highest step .safetensors in checkpoint_dir
            candidates = sorted(checkpoint_dir.glob("ncm_step_*.safetensors"))
            if candidates:
                weights_file = candidates[-1]
                step_str = weights_file.stem.replace("ncm_step_", "")
                candidate_state = checkpoint_dir / f"trainer_state_step_{step_str}.pt"
                if candidate_state.exists():
                    state_file = candidate_state

    if not weights_file.exists():
        print(f"[Resume] No existing checkpoint found in '{checkpoint_dir}'. Starting fresh from step 0.")
        return 0, float("inf"), None

    print(f"[Resume] Loading model weights from '{weights_file}'...")
    state_dict = load_file(str(weights_file), device=str(device))
    model.load_state_dict(state_dict, strict=False)
    model.tie_weights()

    start_step = 0
    best_val_loss = float("inf")
    wandb_run_id: Optional[str] = None

    if state_file.exists():
        print(f"[Resume] Restoring optimizer, LR scheduler, and step state from '{state_file}'...")
        trainer_state = torch.load(state_file, map_location=device, weights_only=False)
        start_step = int(trainer_state.get("step", 0))
        best_val_loss = float(trainer_state.get("best_val_loss", float("inf")))
        wandb_run_id = trainer_state.get("wandb_run_id", None)

        if "optimizer" in trainer_state:
            optimizer.load_state_dict(trainer_state["optimizer"])
        if "scheduler" in trainer_state:
            scheduler.load_state_dict(trainer_state["scheduler"])
        if "window_code_counts" in trainer_state:
            model.window_code_counts.copy_(trainer_state["window_code_counts"].to(device))
        if trainer_state.get("rng_state_cpu") is not None:
            torch.set_rng_state(trainer_state["rng_state_cpu"].cpu())
        if torch.cuda.is_available() and trainer_state.get("rng_state_cuda") is not None:
            try:
                torch.cuda.set_rng_state_all([s.cpu() for s in trainer_state["rng_state_cuda"]])
            except Exception:
                pass
        print(f"[Resume] Successfully resumed at step {start_step} (best_val_loss={best_val_loss:.4f}).")
    else:
        # Infer step from filename if companion trainer_state is absent
        if "ncm_step_" in weights_file.stem:
            try:
                start_step = int(weights_file.stem.replace("ncm_step_", ""))
                for _ in range(start_step):
                    scheduler.step()
            except ValueError:
                pass
        print(f"[Resume] Loaded weights only (no trainer_state.pt found). Starting from step {start_step}.")

    return start_step, best_val_loss, wandb_run_id


@torch.no_grad()
def evaluate_validation(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_eval_batches: int = 25,
) -> dict[str, float]:
    """Evaluates validation text loss, perplexity, and concept loss over held-out batches."""
    was_training = model.training
    model.eval()

    total_text_loss = 0.0
    total_concept_loss = 0.0
    total_vq_loss = 0.0
    count = 0

    for batch_idx, (input_ids, target_ids) in enumerate(val_loader):
        if batch_idx >= max_eval_batches:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(input_ids=input_ids, target_ids=target_ids, return_logits=False)

        total_text_loss += out.text_loss.item() if out.text_loss is not None else 0.0
        total_concept_loss += out.concept_loss.item() if out.concept_loss is not None else 0.0
        total_vq_loss += out.vq_loss.item() if out.vq_loss is not None else 0.0
        count += 1

    if was_training:
        model.train()

    denom = max(1, count)
    avg_text_loss = total_text_loss / denom
    avg_concept_loss = total_concept_loss / denom
    avg_vq_loss = total_vq_loss / denom
    val_ppl = math.exp(min(avg_text_loss, 20.0))

    return {
        "val/text_loss": avg_text_loss,
        "val/perplexity": val_ppl,
        "val/concept_loss": avg_concept_loss,
        "val/vq_loss": avg_vq_loss,
    }


def train(
    bin_path: str = "train.bin",
    val_bin_path: str = "val.bin",
    dataset_tokens: int = 1_000_000_000,
    checkpoint_dir: str = "checkpoints",
    wandb_project: str = "ncm-100m-fineweb-edu",
    wandb_run_name: Optional[str] = "ncm-100m-rvq4-1B",
    batch_size: int = 8,
    grad_accum_steps: int = 4,
    num_workers: int = 2,
    seq_len: int = 1024,
    num_quantizers: int = 4,
    max_steps: int = 2000,
    total_schedule_steps: Optional[int] = 30000,
    warmup_steps: int = 200,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    log_interval: int = 10,
    eval_interval: int = 200,
    save_interval: int = 500,
    resume: bool = False,
    resume_from: Optional[str] = None,
    compile_model: bool = True,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Ensure dataset binary exists (appends up to `dataset_tokens` if needed)
    bin_file = Path(bin_path)
    if not bin_file.exists():
        prepare_fineweb_bin(
            output_bin_path=bin_file,
            val_bin_path=val_bin_path,
            tokenizer_name="gpt2",
            max_tokens=dataset_tokens,
            resume_append=True,
        )

    val_file = Path(val_bin_path)
    if val_file.exists():
        train_dataset = MemmapDataset(bin_path=bin_file, seq_len=seq_len)
        val_dataset = MemmapDataset(bin_path=val_file, seq_len=seq_len)
    else:
        train_dataset = MemmapDataset(bin_path=bin_file, seq_len=seq_len, split="train")
        val_dataset = MemmapDataset(bin_path=bin_file, seq_len=seq_len, split="val")

    use_workers = num_workers > 0
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=use_workers,
        prefetch_factor=2 if use_workers else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    # ~100M Unique Parameters NCM Configuration (ResidualVQ + RoPE + Tied Weights)
    args = ModelArgs(
        vocab_size=50304,
        dim=768,
        concept_dim=384,
        num_concepts=1024,
        num_quantizers=num_quantizers,
        downsample_rate=4,
        planner_layers=5,
        decoder_layers=5,
        num_heads=8,
        max_seq_len=seq_len,
    )

    raw_model = NCMModel(args=args).to(device)
    total_params = raw_model.num_parameters(trainable_only=True)
    print(
        f"Initialized NCMModel (RVQ={num_quantizers}) on {device} with {total_params:,} unique parameters ({total_params / 1e6:.2f}M)"
    )
    print(f"Dataset size: {train_dataset.num_tokens:,} tokens ({len(train_dataset):,} sequences of length {seq_len})")
    flops_per_token = estimate_flops_per_token(raw_model, seq_len=seq_len)

    # 8-bit AdamW optimizer from bitsandbytes to minimize VRAM footprint
    optimizer = bnb.optim.AdamW8bit(
        raw_model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )

    # Cosine Annealing LR Scheduler horizon (`total_schedule_steps` allows multi-session chunked training)
    cosine_horizon = total_schedule_steps if (total_schedule_steps is not None and total_schedule_steps >= max_steps) else max_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=cosine_horizon,
    )

    ckpt_dir = Path(checkpoint_dir)
    start_step = 0
    best_val_loss = float("inf")
    wandb_run_id: Optional[str] = None

    if resume or resume_from is not None:
        start_step, best_val_loss, wandb_run_id = load_training_state(
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_dir=ckpt_dir,
            device=device,
            resume_from=resume_from,
        )

    if start_step >= max_steps:
        print(
            f"[Info] Checkpoint is already at step {start_step} >= target max_steps ({max_steps}). "
            f"Pass a larger `--max-steps` (e.g., --max-steps {start_step + 1000}) to continue training."
        )
        return

    # Compile pure-GPU submodules (Encoder Conv stack, ConceptPlanner, SurfaceDecoder) via Triton PTX
    # while isolating ResidualVQ (@torch.compiler.disable) so MSVC cl.exe is never invoked on Windows.
    model: torch.nn.Module = raw_model
    if compile_model and torch.cuda.is_available():
        try:
            raw_model.compile_submodules(mode="default")
            print("[Info] Enabled torch.compile (Triton PTX) for Encoder Conv, ConceptPlanner, and SurfaceDecoder.")
        except Exception as e:
            print(f"[Info] torch.compile submodule compilation failed ({e}), using eager SDPA.")
            compile_model = False

    # Initialize Weights & Biases logging (seamlessly resumes the exact same WandB chart if `wandb_run_id` exists)
    run = wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        id=wandb_run_id,
        resume="allow" if (resume or resume_from is not None) else None,
        config={
            **vars(args),
            "total_params": total_params,
            "dataset_tokens": train_dataset.num_tokens,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "num_workers": num_workers,
            "effective_batch_tokens": batch_size * grad_accum_steps * seq_len,
            "max_steps": max_steps,
            "total_schedule_steps": cosine_horizon,
            "warmup_steps": warmup_steps,
            "lr": lr,
            "weight_decay": weight_decay,
            "optimizer": "AdamW8bit",
            "precision": "bfloat16",
            "compiled": compile_model,
        },
    )
    active_wandb_id = run.id if run is not None else wandb_run_id
    if run is not None:
        print(f"WandB Run URL: {run.url}", flush=True)

    model.train()
    data_iter = iter(dataloader)
    tokens_per_step = batch_size * seq_len * grad_accum_steps

    # Asynchronous GPU accumulators for zero-sync multi-step logging
    running_total_loss = torch.zeros((), device=device, dtype=torch.float32)
    running_text_loss = torch.zeros((), device=device, dtype=torch.float32)
    running_concept_loss = torch.zeros((), device=device, dtype=torch.float32)
    running_vq_loss = torch.zeros((), device=device, dtype=torch.float32)
    steps_since_log = 0

    t0 = time.perf_counter()
    current_step = start_step
    try:
        for step in range(start_step + 1, max_steps + 1):
            current_step = step
            optimizer.zero_grad(set_to_none=True)

            for _ in range(grad_accum_steps):
                try:
                    input_ids, target_ids = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    input_ids, target_ids = next(data_iter)

                input_ids = input_ids.to(device, non_blocking=True)
                target_ids = target_ids.to(device, non_blocking=True)

                # Mixed-precision bfloat16 forward pass (frees [B, T, V] logits right after loss)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    out = model(input_ids=input_ids, target_ids=target_ids, return_logits=False)
                    assert out.total_loss is not None
                    scaled_loss = out.total_loss / grad_accum_steps

                scaled_loss.backward()

                with torch.no_grad():
                    running_total_loss += out.total_loss.detach() / grad_accum_steps
                    if out.text_loss is not None:
                        running_text_loss += out.text_loss.detach() / grad_accum_steps
                    if out.concept_loss is not None:
                        running_concept_loss += out.concept_loss.detach() / grad_accum_steps
                    if out.vq_loss is not None:
                        running_vq_loss += out.vq_loss.detach() / grad_accum_steps

            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=grad_clip)

            optimizer.step()
            scheduler.step()
            steps_since_log += 1

            if step % log_interval == 0 or step == start_step + 1:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                elapsed = max(t1 - t0, 1e-6)
                tokens_per_sec = (tokens_per_step * steps_since_log) / elapsed
                tflops_per_sec = (tokens_per_sec * flops_per_token) / 1e12

                total_loss_val = running_total_loss.item() / steps_since_log
                text_loss_val = running_text_loss.item() / steps_since_log
                concept_loss_val = running_concept_loss.item() / steps_since_log
                vq_loss_val = running_vq_loss.item() / steps_since_log

                running_total_loss.zero_()
                running_text_loss.zero_()
                running_concept_loss.zero_()
                running_vq_loss.zero_()
                steps_since_log = 0

                current_lr = scheduler.get_last_lr()[0]
                active_codes, window_active_codes, codebook_ppl = raw_model.get_codebook_stats()

                metrics = {
                    "train/step": step,
                    "train/total_loss": total_loss_val,
                    "train/vq_loss": vq_loss_val,
                    "train/concept_loss": concept_loss_val,
                    "train/text_loss": text_loss_val,
                    "train/perplexity": math.exp(min(text_loss_val, 20.0)),
                    "train/active_codes": active_codes,
                    "train/window_active_codes": window_active_codes,
                    "train/codebook_perplexity": codebook_ppl,
                    "train/lr": current_lr,
                    "perf/tokens_per_sec": tokens_per_sec,
                    "perf/tflops": tflops_per_sec,
                }
                wandb.log(metrics, step=step)

                print(
                    f"Step {step:06d}/{max_steps} (sched:{cosine_horizon}) | "
                    f"Loss: {total_loss_val:.4f} (text: {text_loss_val:.4f}, concept: {concept_loss_val:.4f}, vq: {vq_loss_val:.4f}) | "
                    f"BatchCodes: {active_codes}/1024 | WinCodes: {window_active_codes}/1024 (PPL: {codebook_ppl:.1f}) | "
                    f"LR: {current_lr:.2e} | "
                    f"Tok/s: {tokens_per_sec:,.0f} | "
                    f"TFLOP/s: {tflops_per_sec:.2f}",
                    flush=True,
                )
                t0 = time.perf_counter()

            if step % eval_interval == 0 or step == max_steps:
                val_metrics = evaluate_validation(model, val_loader, device=device)
                wandb.log(val_metrics, step=step)
                print(
                    f"[Validation @ Step {step:06d}] "
                    f"Val Text Loss: {val_metrics['val/text_loss']:.4f} | "
                    f"Val PPL: {val_metrics['val/perplexity']:.2f} | "
                    f"Val Concept Loss: {val_metrics['val/concept_loss']:.4f}",
                    flush=True,
                )
                if val_metrics["val/text_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val/text_loss"]
                    best_path = save_safetensors_checkpoint(raw_model, ckpt_dir, filename="ncm_best.safetensors")
                    print(f"[Best Checkpoint] Saved new best model (val_text_loss={best_val_loss:.4f}) to {best_path}")
                t0 = time.perf_counter()

            if step % save_interval == 0 or step == max_steps:
                ckpt_path = save_training_state(
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    checkpoint_dir=ckpt_dir,
                    step=step,
                    best_val_loss=best_val_loss,
                    wandb_run_id=active_wandb_id,
                )
                print(f"[Checkpoint] Saved resumable state & weights at step {step} to {ckpt_path}")

    except KeyboardInterrupt:
        print(f"\n[Interrupt] Caught Ctrl+C at step {current_step}! Saving emergency resumable checkpoint...")
        save_training_state(
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_dir=ckpt_dir,
            step=current_step,
            best_val_loss=best_val_loss,
            wandb_run_id=active_wandb_id,
        )
        print(f"[Interrupt] Saved safely at step {current_step}. Resume anytime with `--resume`.")
    finally:
        wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Next-Concept Modeling (NCM) 100M on FineWeb-Edu (Resumable)")
    parser.add_argument("--bin-path", type=str, default="train.bin")
    parser.add_argument("--val-bin-path", type=str, default="val.bin")
    parser.add_argument("--dataset-tokens", type=int, default=1_000_000_000, help="Target dataset size in tokens (default: 1B)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--wandb-project", type=str, default="ncm-100m-fineweb-edu")
    parser.add_argument("--wandb-run-name", type=str, default="ncm-100m-rvq4-1B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-quantizers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2000, help="Stop training at this step in the current session")
    parser.add_argument(
        "--total-schedule-steps",
        type=int,
        default=30000,
        help="Total steps for Cosine LR schedule across all resumed sessions (30k steps ~= 1B tokens)",
    )
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in checkpoint-dir")
    parser.add_argument("--resume-from", type=str, default=None, help="Optional explicit .safetensors path to resume from")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    train(
        bin_path=cli_args.bin_path,
        val_bin_path=cli_args.val_bin_path,
        dataset_tokens=cli_args.dataset_tokens,
        checkpoint_dir=cli_args.checkpoint_dir,
        wandb_project=cli_args.wandb_project,
        wandb_run_name=cli_args.wandb_run_name,
        batch_size=cli_args.batch_size,
        grad_accum_steps=cli_args.grad_accum_steps,
        num_workers=cli_args.num_workers,
        seq_len=cli_args.seq_len,
        num_quantizers=cli_args.num_quantizers,
        max_steps=cli_args.max_steps,
        total_schedule_steps=cli_args.total_schedule_steps,
        warmup_steps=cli_args.warmup_steps,
        lr=cli_args.lr,
        log_interval=cli_args.log_interval,
        eval_interval=cli_args.eval_interval,
        save_interval=cli_args.save_interval,
        resume=cli_args.resume,
        resume_from=cli_args.resume_from,
    )
