from __future__ import annotations

import argparse
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import AutoTokenizer

from interfaces import ModelArgs
from model import NCMModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text using a pretrained Next-Concept Modeling (NCM) checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_v2/ncm_sft_best.safetensors",
        help="Path to the .safetensors checkpoint file",
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
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="HuggingFace tokenizer name")
    args_cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args_cli.tokenizer, use_fast=True)

    model_args = ModelArgs(
        vocab_size=50304,
        dim=768,
        concept_dim=384,
        num_concepts=1024,
        num_quantizers=args_cli.num_quantizers,
        downsample_rate=4,
        planner_layers=5,
        decoder_layers=5,
        num_heads=8,
        max_seq_len=1024,
    )

    ckpt_path = Path(args_cli.checkpoint)
    if ckpt_path.exists():
        print(f"Loading checkpoint from {ckpt_path} onto {device}...")
        model = NCMModel.from_safetensors(ckpt_path, args=model_args, device=device)
    else:
        print(f"[Warning] Checkpoint '{ckpt_path}' not found; initializing untrained NCMModel for demonstration.")
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

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
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
