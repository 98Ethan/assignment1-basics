"""File processing utilities for BPE training."""

import os
import regex as re
import time
import logging
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import List

logger = logging.getLogger(__name__)

# Pretokenization pattern (shared constant)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_PROCESSES = cpu_count()


def find_chunk_boundaries(file_path: str, num_chunks: int, split_special_token: bytes) -> List[int]:
    """
    Find chunk boundaries based on special tokens.
    
    Args:
        file_path: Path to the input file
        num_chunks: Target number of chunks to split the file into
        split_special_token: Special token as bytes to use as split boundary (e.g., b'<|endoftext|>')
    
    Returns:
        List of byte positions marking chunk boundaries, including start (0) and end positions
    """
    with open(file_path, "rb") as f:
        # Get total file size
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)
        
        chunk_size = file_size // num_chunks
        boundaries = [i * chunk_size for i in range(num_chunks + 1)]
        boundaries[-1] = file_size
        
        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
        
        # Adjust boundaries to special tokens
        for i in range(1, len(boundaries) - 1):
            initial_position = boundaries[i]
            f.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = f.read(mini_chunk_size)  # Read a mini chunk
                
                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    boundaries[i] = file_size
                    break
                
                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    boundaries[i] = initial_position + found_at
                    break
                initial_position += mini_chunk_size
        
        # Make sure all boundaries are unique, but might be fewer than num_chunks
        return sorted(set(boundaries))


def process_file_chunk(args):
    """Worker function: reads file chunk independently and processes to word counter"""
    file_path, start_byte, end_byte, special_token_pattern, chunk_id = args
    
    logger.info(f"Chunk {chunk_id}: Reading file chunk {start_byte}:{end_byte}")
    start_time = time.time()
    
    # Worker reads only its chunk from file
    with open(file_path, "rb") as f:
        f.seek(start_byte)
        chunk_bytes = f.read(end_byte - start_byte)
        chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
    
    # Split on special tokens first to preserve them
    parts = re.split(special_token_pattern, chunk_text)

    words = []
    for part in parts:
        # Apply regular pretokenization pattern
        for match in re.finditer(PAT, part):
            pretoken = match.group()
            words.append(pretoken.encode("utf-8"))
    
    counter = Counter(words)
    elapsed = time.time() - start_time
    
    logger.info(f"Chunk {chunk_id}: Done in {elapsed:.2f}s - {len(counter):,}")
    return counter


def build_word_frequencies(input_path: str, special_tokens: list[str]) -> Counter:
    """
    Build word frequency counter using independent file reads (parallel processing).
    """
    logger.info("Building word frequencies...")
    start_time = time.time()
    
    # Find chunk boundaries
    logger.info("Finding chunk boundaries...")
    boundaries = find_chunk_boundaries(input_path, 100, b"<|endoftext|>")  # More chunks than processes
    
    # Create a pattern to match special tokens
    special_token_pattern = "|".join(re.escape(token) for token in special_tokens)
    
    # Prepare worker arguments with file path and byte boundaries
    worker_args = [
        (input_path, boundaries[i], boundaries[i+1], special_token_pattern, i)
        for i in range(len(boundaries) - 1)
    ]
    
    # Process with workers - each reads file independently
    logger.info(f"Processing {len(worker_args)} chunks with {NUM_PROCESSES} workers...")
    process_start = time.time()
    
    with Pool(processes=NUM_PROCESSES) as pool:
        chunk_counters = pool.map(process_file_chunk, worker_args)
    
    process_time = time.time() - process_start
    logger.info(f"All chunks processed in {process_time:.2f}s")

    # Combine counters efficiently
    logger.info("Combining frequency counters...")
    combine_start = time.time()
    word_freqs = Counter()
    for counter in chunk_counters:
        word_freqs.update(counter)
    combine_time = time.time() - combine_start
    logger.info(f"Combined all counters in {combine_time:.2f}s")
    
    elapsed = time.time() - start_time
    logger.info(f"Built word frequencies in {elapsed:.2f}s")
    logger.info(f"Total words: {sum(word_freqs.values()):,}")
    logger.info(f"Unique words: {len(word_freqs):,}")
    
    return word_freqs