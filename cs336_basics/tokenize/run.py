#!/usr/bin/env python3
"""Run script for tokenizer training and encoding."""

import logging
import numpy as np
from pathlib import Path

from cs336_basics.tokenize.tokenizer import Tokenizer
from cs336_basics.tokenize.io_utils import stream_tokenize_to_bin, save_tokenizer_files, bpe_tokenizer_word_freq

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_bpe(input_path: str, vocab_size: int, output_dir: str, dataset_name: str):
    """Train BPE tokenizer on input text file."""
    logger.info(f"Training BPE tokenizer on {input_path}")
    logger.info(f"Target vocab size: {vocab_size}")
    
    special_tokens = ["<|endoftext|>"]
    
    try:
        vocab, merges = bpe_tokenizer_word_freq(input_path, vocab_size, special_tokens)
        
        logger.info(f"Training completed!")
        logger.info(f"Vocab size: {len(vocab)}")
        logger.info(f"Merges: {len(merges)}")
        
        # Save the trained tokenizer
        logger.info(f"Saving tokenizer files to {output_dir}")
        save_tokenizer_files(vocab, merges, output_dir, dataset_name)
        
        # Show some example merges
        logger.info("First 10 merges:")
        for i, (a, b) in enumerate(merges[:10]):
            try:
                merged_str = (a + b).decode('utf-8', errors='replace')
                a_str = a.decode('utf-8', errors='replace')
                b_str = b.decode('utf-8', errors='replace')
                logger.info(f"{i+1}: '{a_str}' + '{b_str}' -> '{merged_str}'")
            except:
                logger.info(f"{i+1}: {a} + {b}")
                
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def tokenize():
    """Tokenize text files using trained tokenizer."""
    dataset = "TinyStoriesV2-GPT4-valid"

    # Paths
    data_dir = Path("./data")
    output_dir = data_dir / "output"

    # Load your trained tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        str(output_dir / f"{dataset}_vocab.bin"),
        str(output_dir / f"{dataset}_merges.bin"),
    )

    text_file = data_dir / f"{dataset}.txt"
    token_output_file = data_dir / "tokens" / f"{dataset}.bin"

    logger.info(f"Processing file: {text_file}")

    # Create output directory
    token_output_file.parent.mkdir(parents=True, exist_ok=True)

    # Use stream_tokenize_to_bin for efficient processing
    total_tokens = stream_tokenize_to_bin(
        in_path=text_file,
        out_bin=token_output_file,
        tokenizer=tokenizer,  # Pass tokenizer directly (uses encode_iterable)
        flush_every=1_000_000,  # Save every 1M tokens
        dtype=np.uint16,  # Use uint32 if vocab >= 65,536
        encoding="utf-8",
    )

    logger.info(f"✅ Tokenization complete!")
    logger.info(f"📊 Total tokens: {total_tokens:,}")
    logger.info(f"📁 File size: {text_file.stat().st_size / (1024**3):.2f} GB")
    logger.info(f"💾 Saved to: {token_output_file}")


if __name__ == "__main__":
    # Run both training and tokenization
    logger.info("=== TRAINING BPE TOKENIZER ===")
    train_bpe("./data/TinyStoriesV2-GPT4-valid.txt", 10000, "./data/output", "TinyStoriesV2-GPT4-valid")
    
    logger.info("\n=== TOKENIZING FILES ===")
    tokenize()