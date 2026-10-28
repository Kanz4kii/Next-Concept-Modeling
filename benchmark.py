from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from interfaces import ModelArgs
from model import NCMModel


def benchmark_inference(
    ncm_model: NCMModel,
    gpt2_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prompt: str = "Artificial intelligence and machine learning are",
    max_new_tokens: int = 128,
    warmup: int = 2,
    runs: int = 3,
) -> dict[str, dict[str, float]]:
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    print(f"\n[Benchmark] Running Inference Generation Benchmark ({max_new_tokens} tokens)...")

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = ncm_model.generate_text(prompt_ids, max_new_tokens=16, temperature=0.0)
            _ = gpt2_model.generate(prompt_ids, max_new_tokens=16, do_sample=False, pad_token_id=50256)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark NCM v2
    ncm_times = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ncm = ncm_model.generate_text(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                token_temperature=0.0,
                eos_token_id=None,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ncm_times.append(time.perf_counter() - t0)

    avg_ncm_time = sum(ncm_times) / len(ncm_times)
    ncm_tok_per_sec = max_new_tokens / avg_ncm_time

    # Benchmark Standard GPT-2
    gpt2_times = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_gpt = gpt2_model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=50256,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gpt2_times.append(time.perf_counter() - t0)

    avg_gpt2_time = sum(gpt2_times) / len(gpt2_times)
    gpt2_tok_per_sec = max_new_tokens / avg_gpt2_time

    return {
        "NCM v2": {"latency_sec": avg_ncm_time, "tok_per_sec": ncm_tok_per_sec},
        "Standard Transformer (GPT-2)": {"latency_sec": avg_gpt2_time, "tok_per_sec": gpt2_tok_per_sec},
    }


def benchmark_training_throughput(
    ncm_model: NCMModel,
    gpt2_model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 4,
    seq_len: int = 1024,
    steps: int = 10,
) -> dict[str, dict[str, float]]:
    print(f"\n[Benchmark] Running Training Throughput & Peak VRAM Benchmark (B={batch_size}, T={seq_len})...")
    input_ids = torch.randint(0, 50256, (batch_size, seq_len), device=device)
    target_ids = input_ids.clone()

    opt_ncm = torch.optim.AdamW(ncm_model.parameters(), lr=1e-4)
    opt_gpt2 = torch.optim.AdamW(gpt2_model.parameters(), lr=1e-4)

    # 1. Benchmark NCM v2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(2):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = ncm_model(input_ids, target_ids=target_ids, return_logits=False)
            out.total_loss.backward()
        opt_ncm.step()
        opt_ncm.zero_grad(set_to_none=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = ncm_model(input_ids, target_ids=target_ids, return_logits=False)
            out.total_loss.backward()
        opt_ncm.step()
        opt_ncm.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt_ncm = time.perf_counter() - t0
    ncm_train_tok_sec = (steps * batch_size * seq_len) / dt_ncm
    ncm_peak_vram = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0

    # 2. Benchmark Standard Transformer (GPT-2)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(2):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out_g = gpt2_model(input_ids, labels=target_ids)
            out_g.loss.backward()
        opt_gpt2.step()
        opt_gpt2.zero_grad(set_to_none=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out_g = gpt2_model(input_ids, labels=target_ids)
            out_g.loss.backward()
        opt_gpt2.step()
        opt_gpt2.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt_gpt2 = time.perf_counter() - t0
    gpt2_train_tok_sec = (steps * batch_size * seq_len) / dt_gpt2
    gpt2_peak_vram = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0

    return {
        "NCM v2": {"tok_per_sec": ncm_train_tok_sec, "peak_vram_mb": ncm_peak_vram},
        "Standard Transformer (GPT-2)": {"tok_per_sec": gpt2_train_tok_sec, "peak_vram_mb": gpt2_peak_vram},
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    tok_dir = Path("tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir if tok_dir.exists() else "gpt2")

    config_path = Path("config.json")
    ncm_args = ModelArgs.from_json(config_path) if config_path.exists() else ModelArgs()
    ncm_model = NCMModel(args=ncm_args).to(device)

    gpt2_config = GPT2Config(
        vocab_size=50304,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_positions=1024,
    )
    gpt2_model = GPT2LMHeadModel(gpt2_config).to(device)

    print(f"NCM v2 Parameters:                 {ncm_model.num_parameters():,}")
    print(f"Standard Transformer Parameters:    {sum(p.numel() for p in gpt2_model.parameters()):,}")

    train_results = benchmark_training_throughput(ncm_model, gpt2_model, device=device)
    infer_results = benchmark_inference(ncm_model, gpt2_model, tokenizer, device=device)

    print("\n" + "=" * 80)
    print("EMPIRICAL BENCHMARK: NCM v2 vs STANDARD TRANSFORMER (GPT-2 124M)")
    print("=" * 80)
    print(f"{'Model Architecture':<30} | {'Training Tok/s':<15} | {'Peak VRAM':<12} | {'Infer Tok/s':<14}")
    print("-" * 80)
    for name in ["Standard Transformer (GPT-2)", "NCM v2"]:
        tr_speed = f"{train_results[name]['tok_per_sec']:,.0f} tok/s"
        vram = f"{train_results[name]['peak_vram_mb']:,.0f} MB"
        inf_speed = f"{infer_results[name]['tok_per_sec']:.1f} tok/s"
        print(f"{name:<30} | {tr_speed:<15} | {vram:<12} | {inf_speed:<14}")
    print("=" * 80)

    speedup_train = train_results["NCM v2"]["tok_per_sec"] / train_results["Standard Transformer (GPT-2)"]["tok_per_sec"]
    vram_saving = train_results["Standard Transformer (GPT-2)"]["peak_vram_mb"] / max(train_results["NCM v2"]["peak_vram_mb"], 1.0)
    print(f"NCM v2 Training Speedup:     {speedup_train:.2f}x faster throughput")
    print(f"NCM v2 VRAM Efficiency:      {vram_saving:.2f}x less peak VRAM memory")


if __name__ == "__main__":
    main()
