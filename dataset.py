from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def prepare_fineweb_bin(
    output_bin_path: Union[str, Path] = "train.bin",
    val_bin_path: Optional[Union[str, Path]] = "val.bin",
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    subset_name: str = "sample-10BT",
    split: str = "train",
    tokenizer_name: str = "gpt2",
    max_tokens: Optional[int] = 1_000_000_000,
    val_tokens: int = 5_000_000,
    batch_docs: int = 512,
    flush_chunk_tokens: int = 1_048_576,
    resume_append: bool = True,
) -> Path:
    """Streams `HuggingFaceFW/fineweb-edu` (sample-10BT), tokenizes in chunks, and writes flat uint16 to disk.

    Supports seamless continuation (`resume_append=True`): if `train.bin` already exists with fewer than
    `max_tokens` (e.g., 100M tokens), it skips the already-processed tokens and appends only the remaining
    tokens up to `max_tokens` (e.g., 1B tokens).

    Args:
        output_bin_path: Target binary file path (`train.bin`).
        val_bin_path: Optional validation binary file path (`val.bin`).
        dataset_name: HuggingFace dataset identifier.
        subset_name: Dataset configuration/subset (`sample-10BT`).
        split: Dataset split (`train`).
        tokenizer_name: Pretrained tokenizer identifier (must have vocab_size <= 65535 for uint16).
        max_tokens: Cap on total training tokens in `output_bin_path` (default: 1,000,000,000).
        val_tokens: Number of held-out validation tokens written to `val.bin` if `val_bin_path` is set.
        batch_docs: Number of documents tokenized per batch.
        flush_chunk_tokens: Buffer threshold in tokens before flushing uint16 bytes to disk.
        resume_append: If True and `output_bin_path` exists with < `max_tokens`, appends remaining tokens.

    Returns:
        Path to the generated training binary file.
    """
    out_path = Path(output_bin_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    val_path = Path(val_bin_path) if val_bin_path is not None else None

    existing_train_tokens = 0
    if resume_append and out_path.exists():
        existing_train_tokens = os.path.getsize(out_path) // 2
        if max_tokens is not None and existing_train_tokens >= max_tokens:
            print(f"'{out_path}' already contains {existing_train_tokens:,} tokens (>= target {max_tokens:,}).")
            return out_path
        print(
            f"Found existing '{out_path}' with {existing_train_tokens:,} tokens. "
            f"Will skip first {existing_train_tokens:,} tokens and append up to {max_tokens:,} tokens..."
        )

    tok_dir = Path("tokenizer")
    tok_src = tok_dir if (tok_dir.is_dir() and (tok_dir / "tokenizer.json").exists() and tokenizer_name == "gpt2") else tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    tokenizer.model_max_length = int(1e9)
    assert tokenizer.vocab_size <= np.iinfo(np.uint16).max, (
        f"Tokenizer vocab_size ({tokenizer.vocab_size}) exceeds uint16 max (65535)"
    )
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    dataset_stream = load_dataset(
        dataset_name,
        name=subset_name,
        split=split,
        streaming=True,
    )

    token_buffer: list[int] = []
    doc_batch: list[str] = []
    total_train_written = existing_train_tokens
    skipped_train_tokens = 0
    total_val_written = 0
    target_val = val_tokens if (val_path is not None and not val_path.exists()) else 0

    file_mode = "ab" if (resume_append and existing_train_tokens > 0) else "wb"
    f_val = open(val_path, "wb") if target_val > 0 and val_path is not None else None
    try:
        with open(out_path, file_mode) as f_out, tqdm(
            initial=existing_train_tokens,
            total=(max_tokens or 0) + target_val,
            unit="tok",
            unit_scale=True,
            desc=f"Writing {out_path.name}",
        ) as pbar:
            for sample in dataset_stream:
                text = sample.get("text", "")
                if not text:
                    continue
                doc_batch.append(text)

                if len(doc_batch) >= batch_docs:
                    encoded = tokenizer(
                        doc_batch,
                        add_special_tokens=False,
                        return_attention_mask=False,
                    )["input_ids"]
                    for ids in encoded:
                        token_buffer.extend(ids)
                        token_buffer.append(eos_token_id)
                    doc_batch.clear()

                    while len(token_buffer) >= flush_chunk_tokens:
                        chunk = token_buffer[:flush_chunk_tokens]
                        del token_buffer[:flush_chunk_tokens]

                        if f_val is not None and total_val_written < target_val:
                            needed_val = target_val - total_val_written
                            val_chunk = chunk[:needed_val]
                            arr_val = np.asarray(val_chunk, dtype=np.uint16)
                            f_val.write(arr_val.tobytes())
                            total_val_written += len(arr_val)
                            pbar.update(len(arr_val))
                            chunk = chunk[needed_val:]
                            if not chunk:
                                continue

                        # Skip tokens that were already written in a previous run
                        if skipped_train_tokens < existing_train_tokens:
                            need_to_skip = existing_train_tokens - skipped_train_tokens
                            if len(chunk) <= need_to_skip:
                                skipped_train_tokens += len(chunk)
                                continue
                            else:
                                chunk = chunk[need_to_skip:]
                                skipped_train_tokens = existing_train_tokens

                        if max_tokens is not None and total_train_written + len(chunk) > max_tokens:
                            chunk = chunk[: max_tokens - total_train_written]

                        arr = np.asarray(chunk, dtype=np.uint16)
                        f_out.write(arr.tobytes())
                        total_train_written += len(arr)
                        pbar.update(len(arr))

                        if max_tokens is not None and total_train_written >= max_tokens:
                            return out_path

            # Process remaining documents in doc_batch
            if doc_batch and (max_tokens is None or total_train_written < max_tokens):
                encoded = tokenizer(
                    doc_batch,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
                for ids in encoded:
                    token_buffer.extend(ids)
                    token_buffer.append(eos_token_id)
                doc_batch.clear()

            # Flush remaining tokens in buffer
            if token_buffer and (max_tokens is None or total_train_written < max_tokens):
                if skipped_train_tokens < existing_train_tokens:
                    need_to_skip = existing_train_tokens - skipped_train_tokens
                    token_buffer = token_buffer[need_to_skip:]
                    skipped_train_tokens = existing_train_tokens

                if token_buffer:
                    if max_tokens is not None and total_train_written + len(token_buffer) > max_tokens:
                        token_buffer = token_buffer[: max_tokens - total_train_written]
                    arr = np.asarray(token_buffer, dtype=np.uint16)
                    f_out.write(arr.tobytes())
                    total_train_written += len(arr)
                    pbar.update(len(arr))
                    token_buffer.clear()
    finally:
        if f_val is not None:
            f_val.close()

    return out_path


class MemmapDataset(Dataset):
    """Zero-RAM-overhead Dataset reading flat uint16 tokens from disk via `np.memmap`.

    Supports an optional index range (`split='train'` or `split='val'`) over a single `train.bin` file
    if a separate `val.bin` is not present.
    """

    def __init__(
        self,
        bin_path: Union[str, Path] = "train.bin",
        seq_len: int = 2048,
        split: Optional[str] = None,
        val_fraction: float = 0.01,
    ) -> None:
        super().__init__()
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(
                f"Binary token file '{self.bin_path}' not found. Run `prepare_fineweb_bin()` first."
            )
        assert seq_len > 0, f"seq_len must be positive, got {seq_len}"
        self.seq_len = seq_len

        file_size_bytes = os.path.getsize(self.bin_path)
        assert file_size_bytes % 2 == 0, "Corrupted uint16 binary file: byte size is not divisible by 2"
        self.num_tokens = file_size_bytes // 2
        if self.num_tokens <= self.seq_len:
            raise ValueError(
                f"Dataset has only {self.num_tokens} tokens, which is insufficient for seq_len={self.seq_len}"
            )

        total_samples = (self.num_tokens - 1) // self.seq_len
        if split == "train":
            val_count = max(1, int(total_samples * val_fraction))
            self.start_sample = 0
            self.num_samples = total_samples - val_count
        elif split == "val":
            val_count = max(1, int(total_samples * val_fraction))
            self.start_sample = total_samples - val_count
            self.num_samples = val_count
        else:
            self.start_sample = 0
            self.num_samples = total_samples

        self._memmap: Optional[np.memmap] = None

    def __getstate__(self) -> dict:
        """Excludes open `np.memmap` handle when spawning DataLoader workers on Windows (`spawn`)."""
        state = self.__dict__.copy()
        state["_memmap"] = None
        return state

    def _get_memmap(self) -> np.memmap:
        """Lazily opens the memmap per worker process to ensure multi-worker DataLoader safety."""
        if self._memmap is None:
            self._memmap = np.memmap(
                self.bin_path,
                dtype=np.uint16,
                mode="r",
                shape=(self.num_tokens,),
            )
        return self._memmap

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.num_samples}")

        mmap = self._get_memmap()
        real_idx = self.start_sample + idx
        start = real_idx * self.seq_len
        end = start + self.seq_len

        # Fast contiguous conversion from uint16 memmap slice to int64 PyTorch tensor
        input_ids = torch.from_numpy(np.array(mmap[start:end], dtype=np.int64, copy=True))  # [seq_len]
        return input_ids, input_ids.clone()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu flat uint16 binary dataset (1B tokens by default)")
    parser.add_argument("--output-bin", type=str, default="train.bin")
    parser.add_argument("--val-bin", type=str, default="val.bin")
    parser.add_argument("--max-tokens", type=int, default=1_000_000_000, help="Total training tokens (default: 1B)")
    parser.add_argument("--val-tokens", type=int, default=5_000_000, help="Held-out validation tokens (default: 5M)")
    args_cli = parser.parse_args()

    bin_file = Path(args_cli.output_bin)
    prepare_fineweb_bin(
        output_bin_path=bin_file,
        val_bin_path=args_cli.val_bin,
        max_tokens=args_cli.max_tokens,
        val_tokens=args_cli.val_tokens,
        resume_append=True,
    )

    ds_train = MemmapDataset(bin_path=bin_file, seq_len=1024, split="train")
    ds_val = (
        MemmapDataset(bin_path=args_cli.val_bin, seq_len=1024)
        if Path(args_cli.val_bin).exists()
        else MemmapDataset(bin_path=bin_file, seq_len=1024, split="val")
    )
    x, y = ds_train[0]
    print(f"Loaded MemmapDataset: train={len(ds_train):,} seqs, val={len(ds_val):,} seqs ({ds_train.num_tokens:,} total tokens).")
    print(f"Sample shapes: input_ids={tuple(x.shape)}, target_ids={tuple(y.shape)}, dtype={x.dtype}")
