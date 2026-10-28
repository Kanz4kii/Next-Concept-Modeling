from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

from huggingface_hub import HfApi, get_token, login


MODEL_CARD_TEMPLATE = """---
language:
- en
license: mit
tags:
- next-concept-modeling
- ncm
- vector-quantization
- rvq
- semi-autoregressive
- text-generation
- pytorch
datasets:
- HuggingFaceFW/fineweb-edu
- yahma/alpaca-cleaned
metrics:
- perplexity
pipeline_tag: text-generation
---

# Next-Concept Modeling (NCM v2) — 100M SFT

Official model weights and configuration for **Next-Concept Modeling (NCM v2)**: a hierarchical language model architecture combining **Residual Vector Quantization (RVQ-4)** in concept space with **Semi-Autoregressive Surface Realization**.

* **Codebase & Full Implementation:** [GitHub (Kanz4kii/Next-Concept-Modeling)](https://github.com/Kanz4kii/Next-Concept-Modeling)
* **Architecture:** 5-layer ConceptPlanner ($D_c=384$) + 5-layer SurfaceDecoder ($D=768$)
* **Total Active Parameters:** 101.88M
* **Compression Rate:** $r = 4$ tokens per concept (16x high-level self-attention FLOP reduction)
* **Pretraining Data:** 1 Billion tokens of `HuggingFaceFW/fineweb-edu`
* **Instruction Tuning:** 10,000 dialogues from `yahma/alpaca-cleaned`

---

## Benchmark: NCM v2 vs. Standard Transformer (GPT-2 124M)

Measured on a consumer `NVIDIA GeForce RTX 5050` (8GB VRAM) with PyTorch `bfloat16` and FlashAttention SDPA:

| Metric / Regime | Standard Transformer (GPT-2) | NCM v2 (Hierarchical) | NCM Advantage |
| :--- | :---: | :---: | :---: |
| **Model Parameters** | 124.48M (12 layers, $D=768$) | **101.88M** (5 planner + 5 decoder) | **18.2% leaner** |
| **Train Throughput ($B=4, T=1024$)** | 6,667 tok/s (`7,055 MB`) | **23,359 tok/s** (`3,233 MB`) | **`3.50x` faster & `2.18x` less VRAM** |
| **Train Throughput ($B=8, T=1024$)** | 455 tok/s (`13,063 MB` — *RAM paging*) | **27,652 tok/s** (`5,583 MB`) | **`60.8x` faster (zero VRAM spillover)** |
| **Single-Stream Inference ($B=1, 128\text{ tok}$)** | 67.1 tok/s (`1,908 ms`) | **122.4 tok/s** (`1,045 ms`) | **`1.82x` faster latency** |
| **High-Batch Inference ($B=64, 8192\text{ tok}$)** | 3,840.2 tok/s | **7,357.2 tok/s** (`1,113 ms`) | **`1.92x` higher throughput** |
| **High-Level Attention Complexity** | $O(T^2) = 1,048,576$ pairs | $O((T/4)^2) = 65,536$ pairs | **`16x` fewer attention pairs** |

---

## Quick Start & Inference

Clone the official repository to run one-line inference with automatic checkpoint resolution:

```bash
git clone https://github.com/Kanz4kii/Next-Concept-Modeling.git
cd Next-Concept-Modeling
pip install -r requirements.txt
```

### Interactive Chat Mode
```bash
python generate.py \\
  --hf-repo {repo_id} \\
  --chat \\
  --prompt "What are three key principles of effective software engineering?" \\
  --max-new-tokens 128 \\
  --token-temperature 0.5 \\
  --min-p 0.05 \\
  --repetition-penalty 1.25
```

### Python API
```python
import torch
from model import NCMModel
from interfaces import ModelArgs

# Loads model directly from Hugging Face Hub checkpoint
model = NCMModel.from_safetensors("model.safetensors", args=ModelArgs.from_json("config.json"))
model.to("cuda" if torch.cuda.is_available() else "cpu")
```

---

## Citation

```bibtex
@software{{ncm_v2_2026,
  author = {{Kanz4kii}},
  title = {{Next-Concept Modeling (NCM v2): Semi-Autoregressive Hierarchical Language Modeling}},
  year = {{2026}},
  publisher = {{Hugging Face & GitHub}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```
"""


def prepare_staging_dir(
    checkpoint_path: Path,
    config_path: Path,
    tokenizer_dir: Path,
    staging_dir: Path,
    repo_id: str,
) -> None:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Copying weights from {checkpoint_path} -> {staging_dir / 'model.safetensors'}...")
    shutil.copy2(checkpoint_path, staging_dir / "model.safetensors")

    print(f"[2/4] Copying config from {config_path} -> {staging_dir / 'config.json'}...")
    shutil.copy2(config_path, staging_dir / "config.json")

    print(f"[3/4] Copying tokenizer files from {tokenizer_dir}...")
    for item in tokenizer_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, staging_dir / item.name)

    print(f"[4/4] Generating Hugging Face Model Card (README.md)...")
    model_card = MODEL_CARD_TEMPLATE.format(repo_id=repo_id)
    (staging_dir / "README.md").write_text(model_card, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Next-Concept Modeling to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Kanz4kii/Next-Concept-Modeling",
        help="Hugging Face repo id (e.g. Kanz4kii/Next-Concept-Modeling)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_v2/ncm_sft_best.safetensors",
        help="Path to .safetensors checkpoint to upload",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to model config.json",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="tokenizer",
        help="Directory containing tokenizer files",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face write token (optional if already logged in via huggingface-cli login)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Whether to make the Hugging Face repository private",
    )
    args = parser.parse_args()

    token = args.token or get_token()
    if not token:
        print("\n[Auth Required] No Hugging Face token found!")
        print("Please obtain a Write token from: https://huggingface.co/settings/tokens")
        token_input = input("Paste your Hugging Face write token here: ").strip()
        if not token_input:
            print("Error: Token cannot be empty. Aborting.")
            sys.exit(1)
        login(token=token_input)
        token = token_input

    api = HfApi(token=token)

    try:
        user_info = api.whoami()
        print(f"\nAuthenticated on Hugging Face as: @{user_info.get('name', 'Unknown')}")
    except Exception as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)

    ckpt_path = Path(args.checkpoint)
    cfg_path = Path(args.config)
    tok_dir = Path(args.tokenizer_dir)

    if not ckpt_path.exists():
        print(f"Error: Checkpoint file not found: {ckpt_path}")
        sys.exit(1)

    staging_dir = Path("hf_staging")
    print(f"\nPreparing staging directory at '{staging_dir}'...")
    prepare_staging_dir(
        checkpoint_path=ckpt_path,
        config_path=cfg_path,
        tokenizer_dir=tok_dir,
        staging_dir=staging_dir,
        repo_id=args.repo_id,
    )

    print(f"\nCreating / verifying Hugging Face repository: {args.repo_id}...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    print(f"\nUploading files to https://huggingface.co/{args.repo_id}...")
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Initial release of Next-Concept Modeling (NCM v2) weights and configs",
    )

    print("\n" + "=" * 80)
    print("SUCCESSFULLY UPLOADED TO HUGGING FACE HUB!")
    print(f"Model URL: https://huggingface.co/{args.repo_id}")
    print("=" * 80)

    # Clean up staging directory
    if staging_dir.exists():
        shutil.rmtree(staging_dir)


if __name__ == "__main__":
    main()
