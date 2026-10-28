from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import AutoTokenizer

from interfaces import ModelArgs
from model import NCMModel


def resolve_tokenizer(tokenizer_name_or_path: str) -> AutoTokenizer:
    """Loads tokenizer from local 'tokenizer/' directory if available, else from HuggingFace."""
    local_dir = Path("tokenizer")
    if local_dir.is_dir() and (local_dir / "tokenizer.json").exists():
        try:
            return AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        except Exception:
            pass
    return AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)


def resolve_checkpoint(
    checkpoint_arg: Optional[str],
    hf_repo: Optional[str] = None,
) -> Tuple[Optional[Path], bool]:
    """Resolves local checkpoint path or attempts to fetch from Hugging Face Hub if missing."""
    candidates = []
    if checkpoint_arg:
        candidates.append(Path(checkpoint_arg))
    candidates.extend([
        Path("checkpoints_v2/ncm_sft_best.safetensors"),
        Path("checkpoints_v2/ncm_best.safetensors"),
        Path("checkpoints/ncm_sft_best.safetensors"),
        Path("checkpoints/ncm_best.safetensors"),
    ])

    for cand in candidates:
        if cand.is_file():
            return cand, False

    # Check if checkpoint_arg looks like an HF repo ID or hf_repo is explicitly set
    repo_id = hf_repo or (checkpoint_arg if (checkpoint_arg and "/" in checkpoint_arg and not Path(checkpoint_arg).exists()) else None)
    if repo_id:
        try:
            from huggingface_hub import hf_hub_download
            print(f"[Hub] Downloading checkpoint from Hugging Face repository '{repo_id}'...")
            downloaded = hf_hub_download(repo_id=repo_id, filename="ncm_sft_best.safetensors")
            return Path(downloaded), False
        except Exception as e:
            print(f"[Warning] Failed to download from Hugging Face Hub ({e}).")

    return None, True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text using a pretrained Next-Concept Modeling (NCM) checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_v2/ncm_sft_best.safetensors",
        help="Path to the .safetensors checkpoint file or Hugging Face repo ID",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="Optional Hugging Face Hub repository identifier (e.g. 'kanzaki4real/ncm-100m') to auto-download weights",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Artificial intelligence and machine learning are transforming scientific discovery by",
        help="Input text prompt",
    )
    parser.add_argument("--chat", action="store_true", help="Format prompt as an Alpaca instruction-response turn")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.65, help="Concept planner sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K filtering threshold")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus (Top-P) sampling threshold")
    parser.add_argument("--min-p", type=float, default=0.05, help="Min-P sampling threshold (0.05 = keep tokens >= 5 percent of max prob)")
    parser.add_argument("--token-temperature", type=float, default=0.5, help="Surface decoder token temperature (0.0 = greedy)")
    parser.add_argument("--repetition-penalty", type=float, default=1.25, help="Repetition penalty for suppressing token spam")
    parser.add_argument("--repetition-window", type=int, default=64, help="Context window size for repetition penalty")
    parser.add_argument("--ignore-eos", action="store_true", help="Do not stop early on eos_token_id")
    parser.add_argument("--num-quantizers", type=int, default=4, help="Number of ResidualVQ codebooks")
    parser.add_argument("--tokenizer", type=str, default="tokenizer", help="Path to local tokenizer or HF tokenizer name")
    args_cli = parser.parse_args()

    # Device configuration with safe multi-platform fallback (CUDA -> Apple MPS -> CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        amp_device = "cuda"
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        amp_enabled = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        amp_device = "mps"
        amp_dtype = torch.float32
        amp_enabled = False
    else:
        device = torch.device("cpu")
        amp_device = "cpu"
        amp_dtype = torch.float32
        amp_enabled = False

    tokenizer = resolve_tokenizer(args_cli.tokenizer)

    ckpt_path, is_demo = resolve_checkpoint(args_cli.checkpoint, hf_repo=args_cli.hf_repo)
    if ckpt_path is not None and not is_demo:
        print(f"Loading checkpoint from '{ckpt_path}' onto {device}...")
        model = NCMModel.from_safetensors(ckpt_path, device=device)
    else:
        print("=" * 80)
        print("[Notice] Pretrained .safetensors checkpoint not found locally.")
        print("To download the trained weights from Hugging Face:")
        print("    python generate.py --hf-repo kanzaki4real/ncm-100m")
        print("Or place your checkpoint at 'checkpoints_v2/ncm_sft_best.safetensors'.")
        print("Initializing model with random weights for zero-crash pipeline test...")
        print("=" * 80)
        config_path = Path("config.json")
        model_args = ModelArgs.from_json(config_path) if config_path.exists() else ModelArgs()
        model = NCMModel(args=model_args).to(device)

    model.eval()

    prompt_text = args_cli.prompt
    if args_cli.chat and "### Instruction:" not in prompt_text:
        prompt_text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt_text.strip()}\n\n### Response:\n"
        )

    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    print(f"\nPrompt ({prompt_ids.size(1)} tokens):\n{prompt_text}\n")

    with torch.no_grad(), torch.amp.autocast(device_type=amp_device, dtype=amp_dtype, enabled=amp_enabled):
        output_ids = model.generate_text(
            prompt_token_ids=prompt_ids,
            max_new_tokens=args_cli.max_new_tokens,
            temperature=args_cli.temperature,
            top_k=args_cli.top_k,
            top_p=args_cli.top_p,
            min_p=args_cli.min_p,
            token_temperature=args_cli.token_temperature,
            repetition_penalty=args_cli.repetition_penalty,
            repetition_window=args_cli.repetition_window,
            eos_token_id=(None if args_cli.ignore_eos else tokenizer.eos_token_id),
        )

    decoded_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    print("=" * 80)
    print("Generated Output:")
    print("-" * 80)
    print(decoded_text)
    print("=" * 80)


if __name__ == "__main__":
    main()
