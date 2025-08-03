"""Main training interface for BPE tokenizer."""

import time
import logging
from cs336_basics.tokenize.file_utils import build_word_frequencies
from cs336_basics.tokenize.bpe_trainer import WordFrequencyBPE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def bpe_tokenizer_word_freq(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer using word frequency approach.
    
    Args:
        input_path: str Path to a text file with BPE tokenizer training data.
        vocab_size: int A positive integer that defines the maximum final vocabulary size.
        special_tokens: list[str] A list of strings to add to the vocabulary.

    Returns:
        vocab: dict[int, bytes] The tokenizer vocabulary.
        merges: list[tuple[bytes, bytes]] A list of BPE merges.
    """
    logger.info(f"Starting word-frequency BPE tokenizer training on {input_path}")
    logger.info(f"Target vocab size: {vocab_size}, Special tokens: {special_tokens}")
    
    start_total = time.time()
    
    # Build word frequencies from input file
    word_freqs = build_word_frequencies(input_path, special_tokens)
    
    # Train BPE using word frequency approach
    trainer = WordFrequencyBPE(vocab_size, special_tokens)
    vocab, merges = trainer.train(word_freqs)
    
    total_time = time.time() - start_total
    logger.info(f"Total training time: {total_time:.2f}s")
    
    return vocab, merges
