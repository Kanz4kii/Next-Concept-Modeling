from __future__ import annotations

import gc
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from interfaces import ModelArgs
from model import NCMModel

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


def make_optimizer(model: torch.nn.Module, lr: float = 1e-4) -> torch.optim.Optimizer:
    if HAS_BNB and torch.cuda.is_available():
        return bnb.optim.AdamW8bit(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    return torch.optim.AdamW(model.parameters(), lr=lr, fused=torch.cuda.is_available())


def benchmark_single_model_training(
    model_name: str,
    build_fn,
    is_ncm: bool,
    device: torch.device,
    batch_size: int = 8,
    seq_len: int = 1024,
    steps: int = 15,
) -> dict[str, float | str]:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_fn().to(device)
    model.train()
    if is_ncm and torch.cuda.is_available():
        try:
            model.compile_submodules(mode="default")
        except Exception:
            pass

    optimizer = make_optimizer(model)
    input_ids = torch.randint(0, 50256, (batch_size, seq_len), device=device)
    target_ids = input_ids.clone()

    try:
        # Warmup (includes Triton compilation if enabled)
        for _ in range(3):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                if is_ncm:
                    out = model(input_ids, target_ids=target_ids, return_logits=False)
                    loss = out.total_loss
                else:
                    out = model(input_ids, labels=target_ids)
                    loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        for _ in range(steps):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                if is_ncm:
                    out = model(input_ids, target_ids=target_ids, return_logits=False)
                    loss = out.total_loss
                else:
                    out = model(input_ids, labels=target_ids)
                    loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        tok_per_sec = (steps * batch_size * seq_len) / dt
        peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
        status = "OK"
    except torch.OutOfMemoryError:
        tok_per_sec = 0.0
        peak_vram_mb = 8192.0
        status = "CUDA OOM"
    finally:
        del model, optimizer, input_ids, target_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"tok_per_sec": tok_per_sec, "peak_vram_mb": peak_vram_mb, "status": status}


def benchmark_single_model_inference(
    build_fn,
    is_ncm: bool,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int = 1,
    max_new_tokens: int = 128,
    warmup: int = 2,
    runs: int = 5,
) -> dict[str, float]:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = build_fn().to(device)
    model.eval()

    prompt = "Artificial intelligence and machine learning are transforming science by"
    single_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_ids = single_ids.expand(batch_size, -1).contiguous()

    for _ in range(warmup):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            if is_ncm:
                _ = model.generate_text(
                    prompt_ids,
                    max_new_tokens=16,
                    temperature=0.0,
                    token_temperature=0.0,
                    repetition_penalty=1.0,
                    eos_token_id=None,
                )
            else:
                _ = model.generate(
                    prompt_ids,
                    max_new_tokens=16,
                    do_sample=False,
                    pad_token_id=50256,
                )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            if is_ncm:
                _ = model.generate_text(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    token_temperature=0.0,
                    repetition_penalty=1.0,
                    eos_token_id=None,
                )
            else:
                _ = model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=50256,
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg_time = sum(times) / len(times)
    tok_per_sec = (batch_size * max_new_tokens) / avg_time

    del model, prompt_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"latency_sec": avg_time, "tok_per_sec": tok_per_sec}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Benchmarking on device: {device} ({gpu_name})", flush=True)

    tok_dir = Path("tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir if tok_dir.exists() else "gpt2")

    config_path = Path("config.json")
    ncm_args = ModelArgs.from_json(config_path) if config_path.exists() else ModelArgs()

    def build_ncm() -> NCMModel:
        return NCMModel(args=ncm_args)

    def build_gpt2() -> GPT2LMHeadModel:
        gpt2_config = GPT2Config(
            vocab_size=50304,
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_positions=1024,
        )
        return GPT2LMHeadModel(gpt2_config)

    tmp_ncm = build_ncm()
    tmp_gpt = build_gpt2()
    print(f"NCM v2 Parameters:                 {tmp_ncm.num_parameters():,}", flush=True)
    print(f"Standard Transformer Parameters:    {sum(p.numel() for p in tmp_gpt.parameters()):,}", flush=True)
    del tmp_ncm, tmp_gpt

    print("\n[1/3] Running Training Benchmark at B=4, T=1024 (Both models fit in 8GB VRAM)...", flush=True)
    train_ncm_b4 = benchmark_single_model_training("NCM v2", build_ncm, True, device, batch_size=4, seq_len=1024, steps=10)
    train_gpt_b4 = benchmark_single_model_training("GPT-2", build_gpt2, False, device, batch_size=4, seq_len=1024, steps=10)

    print("[2/3] Running Training Benchmark at B=8, T=1024 (High-throughput batch regime)...", flush=True)
    train_ncm_b8 = benchmark_single_model_training("NCM v2", build_ncm, True, device, batch_size=8, seq_len=1024, steps=10)
    train_gpt_b8 = benchmark_single_model_training("GPT-2", build_gpt2, False, device, batch_size=8, seq_len=1024, steps=3)

    print("[3/3] Running Inference Benchmark (B=1 Latency & B=16 Batched Throughput)...", flush=True)
    infer_ncm_b1 = benchmark_single_model_inference(build_ncm, True, tokenizer, device, batch_size=1, max_new_tokens=128)
    infer_gpt_b1 = benchmark_single_model_inference(build_gpt2, False, tokenizer, device, batch_size=1, max_new_tokens=128)

    infer_ncm_b16 = benchmark_single_model_inference(build_ncm, True, tokenizer, device, batch_size=16, max_new_tokens=128)
    infer_gpt_b16 = benchmark_single_model_inference(build_gpt2, False, tokenizer, device, batch_size=16, max_new_tokens=128)

    print("\n" + "=" * 95)
    print(f"EMPIRICAL BENCHMARK ON {gpu_name.upper()} (PyTorch bfloat16 + SDPA)")
    print("=" * 95)
    print(
        f"{'Metric':<38} | {'Standard Transformer (GPT-2)':<26} | {'NCM v2 (Hierarchical)':<24}"
    )
    print("-" * 95)
    print(
        f"{'Train Throughput (B=4, T=1024)':<38} | "
        f"{train_gpt_b4['tok_per_sec']:,.0f} tok/s ({train_gpt_b4['peak_vram_mb']:,.0f} MB) | "
        f"{train_ncm_b4['tok_per_sec']:,.0f} tok/s ({train_ncm_b4['peak_vram_mb']:,.0f} MB)"
    )
    gpt_b8_str = (
        "CUDA OOM (>8 GB)"
        if train_gpt_b8["status"] != "OK"
        else f"{train_gpt_b8['tok_per_sec']:,.0f} tok/s ({train_gpt_b8['peak_vram_mb']:,.0f} MB)"
    )
    print(
        f"{'Train Throughput (B=8, T=1024)':<38} | "
        f"{gpt_b8_str:<26} | "
        f"{train_ncm_b8['tok_per_sec']:,.0f} tok/s ({train_ncm_b8['peak_vram_mb']:,.0f} MB)"
    )
    print(
        f"{'Inference Speed (B=1, 128 tok)':<38} | "
        f"{infer_gpt_b1['tok_per_sec']:,.1f} tok/s ({infer_gpt_b1['latency_sec']*1000:.0f} ms)     | "
        f"{infer_ncm_b1['tok_per_sec']:,.1f} tok/s ({infer_ncm_b1['latency_sec']*1000:.0f} ms)"
    )
    print(
        f"{'Batched Inference (B=16, 2048 tok)':<38} | "
        f"{infer_gpt_b16['tok_per_sec']:,.1f} tok/s               | "
        f"{infer_ncm_b16['tok_per_sec']:,.1f} tok/s"
    )
    print("=" * 95, flush=True)


if __name__ == "__main__":
    main()
