from __future__ import annotations

from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


class AlpacaSFTDataset(Dataset):
    """Preprocessed and tokenized SFT Dataset for Next-Concept Modeling."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_len: int = 512,
        split_name: str = "train[:10000]",
        val_samples: int = 500,
        is_val: bool = False,
    ) -> None:
        super().__init__()
        assert max_seq_len % 4 == 0, f"max_seq_len ({max_seq_len}) must be divisible by 4"
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 50256

        cache_file = Path("checkpoints") / f"alpaca_sft_{'val' if is_val else 'train'}_{val_samples}_{max_seq_len}.pt"
        if cache_file.exists():
            print(f"[Dataset] Loading cached tokenized dataset from '{cache_file}'...", flush=True)
            self.examples = torch.load(cache_file, weights_only=True)
            print(f"[Dataset] Loaded {len(self.examples)} examples from cache.", flush=True)
            return

        print(f"Loading SFT dataset 'yahma/alpaca-cleaned' ({split_name})...", flush=True)
        raw_ds = load_dataset("yahma/alpaca-cleaned", split=split_name)

        if is_val:
            samples = raw_ds.select(range(len(raw_ds) - val_samples, len(raw_ds)))
        else:
            samples = raw_ds.select(range(0, len(raw_ds) - val_samples))

        print(f"Tokenizing {len(samples)} examples for {'validation' if is_val else 'training'} (max_seq_len={max_seq_len})...", flush=True)
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []

        prompt_input_tmpl = PROMPT_DICT["prompt_input"]
        prompt_no_input_tmpl = PROMPT_DICT["prompt_no_input"]

        for ex in samples:
            inst = ex.get("instruction", "").strip()
            inp = ex.get("input", "").strip()
            out = ex.get("output", "").strip()

            if inp:
                p_text = prompt_input_tmpl.format(instruction=inst, input=inp)
            else:
                p_text = prompt_no_input_tmpl.format(instruction=inst)

            r_text = out + tokenizer.eos_token

            p_ids = tokenizer.encode(p_text, add_special_tokens=False)
            r_ids = tokenizer.encode(r_text, add_special_tokens=False)

            if len(p_ids) >= max_seq_len - 8:
                continue

            max_r_len = max_seq_len - len(p_ids)
            r_ids = r_ids[:max_r_len]
            if not r_ids:
                continue

            input_ids = p_ids + r_ids
            labels = [-100] * len(p_ids) + r_ids

            pad_len = max_seq_len - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [self.eos_token_id] * pad_len
                labels = labels + [-100] * pad_len

            self.examples.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            ))

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.examples, cache_file)
        print(f"Successfully prepared & cached {len(self.examples)} valid examples to '{cache_file}'.", flush=True)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[idx]


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("gpt2")
    ds = AlpacaSFTDataset(tok, max_seq_len=512, split_name="train[:1000]", val_samples=100, is_val=False)
    x, y = ds[0]
    print(f"Sample 0 input_ids shape: {x.shape}, labels shape: {y.shape}")
    print(f"Number of response tokens to train on: {(y != -100).sum().item()}")
