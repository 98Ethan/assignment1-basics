"""I/O utilities for saving and loading tokenizer files."""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def save_tokenizer_files(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], 
                        output_dir: str = "./data/output", prefix: str = "tokenizer") -> None:
    """
    Save vocabulary and merges to binary format files.
    
    Args:
        vocab: Dictionary mapping token IDs to their byte representations
        merges: List of merge rules as (bytes, bytes) tuples
        output_dir: Directory to save files
        prefix: Prefix for output filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save vocabulary in binary format
    vocab_bin_path = os.path.join(output_dir, f"{prefix}_vocab.bin")
    with open(vocab_bin_path, 'wb') as f:
        # Write vocabulary size
        f.write(len(vocab).to_bytes(4, byteorder='little'))
        
        # Write each token: <id(4 bytes)><length(4 bytes)><token content(bytes)>
        for token_id, token in vocab.items():
            f.write(token_id.to_bytes(4, byteorder='little'))
            f.write(len(token).to_bytes(4, byteorder='little'))
            f.write(token)
    
    # Save merges in binary format
    merges_bin_path = os.path.join(output_dir, f"{prefix}_merges.bin")
    with open(merges_bin_path, 'wb') as f:
        # Write number of merges
        f.write(len(merges).to_bytes(4, byteorder='little'))
        
        # Write each merge: <first_length(4 bytes)><first_content><second_length(4 bytes)><second_content>
        for first, second in merges:
            f.write(len(first).to_bytes(4, byteorder='little'))
            f.write(first)
            f.write(len(second).to_bytes(4, byteorder='little'))
            f.write(second)
    
    logger.info(f"Saved tokenizer files:")
    logger.info(f"  Vocab: {vocab_bin_path}")
    logger.info(f"  Merges: {merges_bin_path}")


def stream_tokenize_to_bin(
    in_path: Path,
    out_bin: Path,
    tokenizer,  # Tokenizer instance with encode_iterable method
    *,
    flush_every: int = 1_000_000,  # tokens to buffer before writing
    dtype: np.dtype = np.uint16,  # use np.uint32 if vocab ≥ 65,536
    encoding: str = "utf-8",
) -> int:
    """
    Tokenize a large text file line-by-line and append token IDs to a flat binary file.
    Uses tokenizer.encode_iterable for memory-efficient processing.
    Returns total token count written.
    """
    buf = []
    total = 0
    logger.info(f"Starting tokenization: {in_path} → {out_bin}")
    with (
        open(in_path, encoding=encoding) as fin,
        open(out_bin, "ab", buffering=1024 * 1024) as fbin,
    ):
        # Use encode_iterable for memory-efficient tokenization
        for token_id in tokenizer.encode_iterable(fin):
            buf.append(token_id)
            total += 1
            # print(f"Processed {total:,} tokens...", end="\r")  # Show progress

            if len(buf) >= flush_every:
                np.asarray(buf, dtype=dtype).tofile(fbin)
                logger.info(f"Flushed {len(buf):,} tokens to disk (total: {total:,})")
                buf.clear()

        if buf:
            np.asarray(buf, dtype=dtype).tofile(fbin)
            logger.info(f"Final flush: {len(buf):,} tokens to disk (total: {total:,})")

    logger.info(f"Tokenization complete. Total tokens written: {total:,}")
    return total


def load_token_file(file_path: Union[str, Path]) -> np.ndarray:
    """Load tokenized file as numpy array."""
    return np.load(file_path)


def load_token_file_mmap(file_path: Union[str, Path], dype=np.uint16) -> np.ndarray:
    """Load tokenized file as memory-mapped array (doesn't load into RAM)."""
    return np.memmap(file_path, dtype=dype, mode='r')