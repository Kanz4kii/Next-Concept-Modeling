from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from interfaces import ModelArgs
from model import NCMModel
from sft_dataset import AlpacaSFTDataset
from train import save_safetensors_checkpoint


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float = 1e-5,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        base = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale to range [min_lr / base_lr, 1.0]
        return max(min_lr / 1e-4, base)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_sft(
    model: NCMModel,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 25,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_text_loss = 0.0
    total_concept_loss = 0.0
    num_batches = 0

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i, (input_ids, target_ids) in enumerate(val_loader):
            if i >= max_batches:
                break
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)

            out = model(input_ids=input_ids, target_ids=target_ids, return_logits=False)
            if out.total_loss is not None:
                total_loss += out.total_loss.item()
            if out.text_loss is not None:
                total_text_loss += out.text_loss.item()
            if out.concept_loss is not None:
                total_concept_loss += out.concept_loss.item()
            num_batches += 1

    model.train()
    model.tokenizer.vq.eval()
    if num_batches == 0:
        return 0.0, 0.0, 0.0
    return (
        total_loss / num_batches,
        total_text_loss / num_batches,
        total_concept_loss / num_batches,
    )


def sample_test_generation(
    model: NCMModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prompt: str = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat are three tips for staying healthy?\n\n### Response:\n",
) -> str:
    was_training = model.training
    model.eval()

    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate_text(
            prompt_token_ids=prompt_ids,
            max_new_tokens=64,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            min_p=0.05,
            token_temperature=0.55,
            repetition_penalty=1.25,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    if was_training:
        model.train()
        model.tokenizer.vq.eval()
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) for NCM on 10k Alpaca instructions")
    parser.add_argument("--base-checkpoint", type=str, default="checkpoints/ncm_step_030000.safetensors")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--freeze-codebook", action="store_true", default=True)
    parser.add_argument("--use-compile", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load SFT Datasets
    train_dataset = AlpacaSFTDataset(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        split_name="train[:10000]",
        val_samples=args.val_samples,
        is_val=False,
    )
    val_dataset = AlpacaSFTDataset(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        split_name="train[:10000]",
        val_samples=args.val_samples,
        is_val=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    # 2. Initialize Model and Load Pretrained Weights
    model_args = ModelArgs(
        vocab_size=50304,
        dim=768,
        concept_dim=384,
        num_concepts=1024,
        num_quantizers=4,
        downsample_rate=4,
        planner_layers=5,
        decoder_layers=5,
        num_heads=8,
        max_seq_len=1024,
    )

    base_ckpt = Path(args.base_checkpoint)
    assert base_ckpt.exists(), f"Base checkpoint '{base_ckpt}' not found!"
    print(f"Loading pretrained base model weights from '{base_ckpt}'...")
    model = NCMModel.from_safetensors(base_ckpt, args=model_args, device=device)

    # Freeze VQ Tokenizer codebooks to ensure concepts stay rock solid
    if args.freeze_codebook:
        print("[SFT] Freezing VQ Tokenizer codebooks (ResidualVQ) and locking EMA buffers in eval mode...")
        model.tokenizer.requires_grad_(False)
        model.tokenizer.vq.eval()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {num_trainable:,} / {model.num_parameters():,}")

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True if torch.cuda.is_available() else False,
    )

    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
        min_lr=args.min_lr,
    )

    # Optional torch.compile
    raw_model = model
    if args.use_compile and hasattr(torch, "compile"):
        try:
            print("[Info] Compiling submodules for fast SFT throughput...")
            model.compile_submodules(mode="default")
        except Exception as e:
            print(f"[Warning] torch.compile skipped: {e}")

    # Initial Validation
    print("\nRunning pre-SFT initial validation...", flush=True)
    init_val_loss, init_text_loss, init_concept_loss = evaluate_sft(raw_model, val_loader, device)
    init_ppl = math.exp(min(init_text_loss, 20.0))
    print(f"Initial Val Loss: {init_val_loss:.4f} | Text Loss: {init_text_loss:.4f} (PPL: {init_ppl:.2f}) | Concept Loss: {init_concept_loss:.4f}", flush=True)

    # 4. Training Loop
    global_step = 0
    best_val_loss = init_val_loss
    model.train()
    if args.freeze_codebook:
        model.tokenizer.vq.eval()
    optimizer.zero_grad(set_to_none=True)

    accum_loss = 0.0
    accum_text_loss = 0.0
    accum_concept_loss = 0.0
    steps_since_log = 0
    step_tokens = 0
    t0 = time.time()

    print(f"\nStarting SFT training across {args.epochs} epochs (~{total_steps} total optimization steps)...", flush=True)

    for epoch in range(1, args.epochs + 1):
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, target_ids=target_ids, return_logits=False)
                loss = out.total_loss / args.grad_accum_steps

            loss.backward()

            accum_loss += out.total_loss.item() / args.grad_accum_steps
            if out.text_loss is not None:
                accum_text_loss += out.text_loss.item() / args.grad_accum_steps
            if out.concept_loss is not None:
                accum_concept_loss += out.concept_loss.item() / args.grad_accum_steps
            step_tokens += input_ids.numel()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                steps_since_log += 1

                # Logging
                if global_step % 10 == 0 or global_step == 1:
                    dt = time.time() - t0
                    tok_sec = step_tokens / max(dt, 1e-4)
                    cur_lr = scheduler.get_last_lr()[0]
                    avg_loss = accum_loss / max(steps_since_log, 1)
                    avg_text = accum_text_loss / max(steps_since_log, 1)
                    avg_concept = accum_concept_loss / max(steps_since_log, 1)
                    text_ppl = math.exp(min(avg_text, 20.0))
                    print(
                        f"Epoch {epoch}/{args.epochs} | Step {global_step:04d}/{total_steps:04d} | "
                        f"Loss: {avg_loss:.4f} (text: {avg_text:.4f}, PPL: {text_ppl:.1f}, "
                        f"concept: {avg_concept:.4f}) | LR: {cur_lr:.2e} | Speed: {tok_sec:,.0f} tok/s",
                        flush=True,
                    )
                    accum_loss = 0.0
                    accum_text_loss = 0.0
                    accum_concept_loss = 0.0
                    steps_since_log = 0
                    step_tokens = 0
                    t0 = time.time()

                # Periodic Evaluation & Checkpoint
                if global_step % args.eval_interval == 0 or global_step == total_steps:
                    val_loss, val_text, val_concept = evaluate_sft(raw_model, val_loader, device)
                    val_ppl = math.exp(min(val_text, 20.0))
                    print(
                        f"\n[Validation @ SFT Step {global_step:04d}] Val Loss: {val_loss:.4f} | "
                        f"Text Loss: {val_text:.4f} | Val PPL: {val_ppl:.2f} | Concept Loss: {val_concept:.4f}",
                        flush=True,
                    )

                    # Sample generation
                    sample_output = sample_test_generation(raw_model, tokenizer, device)
                    print("Sample output:", flush=True)
                    print("-" * 50, flush=True)
                    print(sample_output, flush=True)
                    print("-" * 50 + "\n", flush=True)

                    # Save latest
                    save_safetensors_checkpoint(raw_model, out_dir, "ncm_sft_latest.safetensors")

                    # Save best
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = save_safetensors_checkpoint(raw_model, out_dir, "ncm_sft_best.safetensors")
                        print(f"==> New best SFT model saved to '{best_path}' (val_loss: {best_val_loss:.4f})", flush=True)

                    t0 = time.time()

    print(f"\n[SFT Complete] Finished all {args.epochs} epochs ({global_step} steps).", flush=True)
    final_path = save_safetensors_checkpoint(raw_model, out_dir, "ncm_sft_final.safetensors")
    print(f"Final SFT checkpoint saved to '{final_path}'.", flush=True)


if __name__ == "__main__":
    main()
