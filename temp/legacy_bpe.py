import os
import regex as re
import time
import logging
from typing import BinaryIO
from itertools import chain

from multiprocessing import Pool, cpu_count

# Set up logging
# logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_PROCESSES = cpu_count()  # Number of processes to use for parallel processing
REMOVED_TOKEN = -1  # Special marker for removed tokens

# ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
# It may be useful to interactively split some text with this pre-tokenizer to get a better sense of its behavior:


# the end-of-sequence string <|endoftext|> should always be preserved as a single token (i.e., a single integer ID)


#   Always split at the beginning of special tokens:
#   Chunk 1: "...some text "
#   Chunk 2: "<|endoftext|> more text..."

#   This ensures:
#   1. Special tokens remain intact and get tokenized correctly
#   2. Document boundaries are preserved (since <|endoftext|> marks document ends)
#   3. No merging occurs across document boundaries during BPE training


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    1. Roughly divide the file: Split a large file into approximately equal-sized chunks for parallel processing
    2. Respect split special token boundaries: Ensure chunks don't split in the middle of special tokens like <|endoftext|>

    Each chunk may still contain split_special_token, meaning more than one documents

    Args:
        file: Binary file object opened for reading
        desired_num_chunks: Target number of chunks to split the file into
        split_special_token: Special token as bytes to use as split boundary (e.g., b'<|endoftext|>')

    Returns:
        List of byte positions marking chunk boundaries, including start (0) and end positions
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(file_path: str, start: int, end: int, special_token_pattern: str) -> list[int]:
    """
    Process a single chunk of the file for pretokenization.
    
    Reads a chunk of text from the specified file range and splits it into pretokens
    using regex patterns while preserving special tokens, then converts to byte values.
    
    Args:
        file_path: str Path to the input file to read from
        start: int Starting byte position in the file
        end: int Ending byte position in the file  
        special_token_pattern: str Regex pattern to match special tokens
        
    Returns:
        list[int] List of byte values (0-255) from all pretokens, ready for BPE pair counting
    """
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # Split on special tokens first to preserve them
    parts = re.split(special_token_pattern, chunk)

    byte_values = []
    for part in parts:
        # Apply regular pretokenization pattern
        for match in re.finditer(PAT, part):
            pretoken = match.group()
            byte_values.extend(list(pretoken.encode("utf-8")))

    return byte_values


def build_pair_index(tokens: list[int]) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], set[int]]]:
    """
    Build initial pair index - only called once before any merges.
    
    Args:
        tokens: list[int] Token sequence (clean, no removed markers)
        
    Returns:
        tuple of (pair_counts, pair_positions) where:
        - pair_counts: dict mapping pairs to their frequency
        - pair_positions: dict mapping pairs to set of positions where they occur
    """
    pair_counts = {}
    pair_positions = {}
    
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if pair not in pair_positions:
            pair_positions[pair] = set()
        pair_positions[pair].add(i)
    
    return pair_counts, pair_positions


def apply_merge(tokens: list[int], pair_to_merge: tuple[int, int], new_token_id: int, 
               pair_counts: dict[tuple[int, int], int], 
               pair_positions: dict[tuple[int, int], set[int]]) -> None:
    """
    Apply a BPE merge by marking second token as REMOVED_TOKEN and incrementally updating indices.
    
    Args:
        tokens: list[int] Current token sequence (modified in place)
        pair_to_merge: tuple[int, int] The pair to merge
        new_token_id: int New token ID to replace the pair
        pair_counts: dict Pair frequency counts to update
        pair_positions: dict Pair positions to update
    """
    if pair_to_merge not in pair_positions:
        return
    
    positions_to_merge = list(pair_positions[pair_to_merge])
    
    # Apply merges and update affected pairs
    for pos in positions_to_merge:
        if tokens[pos] == REMOVED_TOKEN: # possible when overlapping pairs: e.g. 5, 5, 5, and merge (5, 5)
            continue
            
        # Find next non-removed token
        next_pos = pos + 1
        while next_pos < len(tokens) and tokens[next_pos] == REMOVED_TOKEN:
            next_pos += 1
            
        if next_pos < len(tokens) and (tokens[pos], tokens[next_pos]) == pair_to_merge:
            # Update counts for affected neighboring pairs before merge
            _update_indices_for_merge(tokens, pos, next_pos, new_token_id, pair_counts, pair_positions)
            
            # Perform the merge: replace first token, mark second as removed
            tokens[pos] = new_token_id
            tokens[next_pos] = REMOVED_TOKEN
    
    # Remove the merged pair from indices
    if pair_to_merge in pair_counts:
        del pair_counts[pair_to_merge]
    if pair_to_merge in pair_positions:
        del pair_positions[pair_to_merge]


def _update_indices_for_merge(tokens: list[int], pos: int, next_pos: int, new_token_id: int,
                             pair_counts: dict[tuple[int, int], int], 
                             pair_positions: dict[tuple[int, int], set[int]]) -> None:
    """
    Incrementally update pair indices when performing a merge.
    """
    # Find previous non-removed token
    prev_pos = pos - 1
    while prev_pos >= 0 and tokens[prev_pos] == REMOVED_TOKEN:
        prev_pos -= 1
        
    # Find next non-removed token after next_pos
    after_pos = next_pos + 1
    while after_pos < len(tokens) and tokens[after_pos] == REMOVED_TOKEN:
        after_pos += 1
    
    # Remove old left pair (prev_token, tokens[pos])
    if prev_pos >= 0:
        left_pair = (tokens[prev_pos], tokens[pos])
        if left_pair in pair_counts:
            pair_counts[left_pair] -= 1
            pair_positions[left_pair].discard(prev_pos)
            if pair_counts[left_pair] == 0:
                del pair_counts[left_pair]
                del pair_positions[left_pair]
    
    # Remove old right pair (tokens[next_pos], after_token)
    if after_pos < len(tokens):
        right_pair = (tokens[next_pos], tokens[after_pos])
        if right_pair in pair_counts:
            pair_counts[right_pair] -= 1
            pair_positions[right_pair].discard(next_pos)
            if pair_counts[right_pair] == 0:
                del pair_counts[right_pair]
                del pair_positions[right_pair]
    
    # Add new left pair (prev_token, new_token_id)
    if prev_pos >= 0:
        new_left_pair = (tokens[prev_pos], new_token_id)
        pair_counts[new_left_pair] = pair_counts.get(new_left_pair, 0) + 1
        if new_left_pair not in pair_positions:
            pair_positions[new_left_pair] = set()
        pair_positions[new_left_pair].add(prev_pos)
    
    # Add new right pair (new_token_id, after_token)
    if after_pos < len(tokens):
        new_right_pair = (new_token_id, tokens[after_pos])
        pair_counts[new_right_pair] = pair_counts.get(new_right_pair, 0) + 1
        if new_right_pair not in pair_positions:
            pair_positions[new_right_pair] = set()
        pair_positions[new_right_pair].add(pos)


def train_bpe_merges(tokens: list[int], num_merges: int, 
                    initial_vocab: dict[int, bytes]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train BPE merges using indexed approach for efficiency.
    
    Args:
        tokens: list[int] Flattened sequence of all tokens
        initial_vocab: dict[int, bytes] Initial vocabulary including bytes and special tokens
        
    Returns:
        tuple of (vocab, merges) where:
        - vocab: dict mapping token IDs to their byte representations
        - merges: list of merge rules as (bytes, bytes) tuples
    """
    vocab = initial_vocab
    merges = []
    next_token_id = len(initial_vocab)
    
    # Build initial pair index
    logger.info("Building initial pair index...")
    index_start = time.time()
    pair_counts, pair_positions = build_pair_index(tokens)
    index_time = time.time() - index_start
    logger.info(f"Built pair index in {index_time:.2f}s, found {len(pair_counts)} unique pairs")
    
    # Perform BPE merges
    logger.info(f"Performing {num_merges} BPE merges...")
    merge_start = time.time()
    
    for merge_idx in range(num_merges):
        if not pair_counts:
            logger.info(f"No more pairs to merge, stopping at {merge_idx} merges")
            break
            
        # Find most frequent pair
        top_pair = max(pair_counts, key=pair_counts.get)
        top_count = pair_counts[top_pair]
        
        # Apply merge and update indices incrementally
        apply_merge(tokens, top_pair, next_token_id, pair_counts, pair_positions)
        
        # Record the merge
        token1_bytes = vocab[top_pair[0]]
        token2_bytes = vocab[top_pair[1]]
        merges.append((token1_bytes, token2_bytes))
        vocab[next_token_id] = token1_bytes + token2_bytes
        
        # Log progress every 50 merges
        if (merge_idx + 1) % 50 == 0 or merge_idx < 10:
            elapsed = time.time() - merge_start
            avg_time = elapsed / (merge_idx + 1)
            eta = avg_time * (num_merges - merge_idx - 1)
            try:
                merged_str = (token1_bytes + token2_bytes).decode('utf-8', errors='replace')
                logger.info(f"Merge {merge_idx+1}/{num_merges}: '{merged_str}' (count={top_count}) - {elapsed:.1f}s elapsed, ETA {eta:.1f}s")
            except:
                logger.info(f"Merge {merge_idx+1}/{num_merges}: {token1_bytes}+{token2_bytes} (count={top_count}) - {elapsed:.1f}s elapsed, ETA {eta:.1f}s")
        
        next_token_id += 1
    
    total_merge_time = time.time() - merge_start
    logger.info(f"Completed {len(merges)} merges in {total_merge_time:.2f}s (avg {total_merge_time/len(merges):.3f}s per merge)")
    
    return vocab, merges
    

def bpe_tokenizer(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given input file.

    Args:
        input_path: str Path to a text file with BPE tokenizer training data.
        vocab_size: int A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.

    Returns:
        vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
        merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.
    """
    logger.info(f"Starting BPE tokenizer training on {input_path}")
    logger.info(f"Target vocab size: {vocab_size}, Special tokens: {special_tokens}")
    
    start_total = time.time()
    
    # Get chunk boundaries
    logger.info("Finding chunk boundaries...")
    boundary_start = time.time()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, NUM_PROCESSES, "<|endoftext|>".encode("utf-8")
        )
    boundary_time = time.time() - boundary_start
    logger.info(f"Found {len(boundaries)-1} chunks in {boundary_time:.2f}s")

    # Create a pattern to match special tokens
    special_token_pattern = "|".join(re.escape(token) for token in special_tokens)

    # Process in parallel using multiprocessing
    logger.info(f"Starting pretokenization with {cpu_count()} processes...")
    pretokenize_start = time.time()
    
    with Pool(processes=NUM_PROCESSES) as pool:
        # Prepare arguments for each chunk
        chunk_args = [
            (input_path, start, end, special_token_pattern)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        # Process all chunks in parallel using starmap
        byte_chunks = pool.starmap(pretokenize, chunk_args)

    # Flatten all byte chunks into one sequence for indexed BPE
    tokens: list[int] = list(chain.from_iterable(byte_chunks))
    pretokenize_time = time.time() - pretokenize_start
    
    logger.info(f"Pretokenization completed in {pretokenize_time:.2f}s")
    logger.info(f"Total tokens: {len(tokens):,}")
    
    # Initialize vocabulary with special tokens
    logger.info("Initializing vocabulary...")
    initial_vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256
    # Add special tokens to initial vocabulary
    for token in special_tokens:
        initial_vocab[next_token_id] = token.encode('utf-8')
        next_token_id += 1
    
    logger.info(f"Initial vocab size: {len(initial_vocab)} (256 bytes + {len(special_tokens)} special tokens)")
    
    # Train BPE merges
    num_merges = vocab_size - len(initial_vocab)
    logger.info(f"Starting BPE training with {num_merges} merges...")
    merge_start = time.time()
    
    vocab, merges = train_bpe_merges(tokens, num_merges, initial_vocab)
    
    merge_time = time.time() - merge_start
    total_time = time.time() - start_total
    
    logger.info(f"BPE training completed in {merge_time:.2f}s")
    logger.info(f"Total training time: {total_time:.2f}s")
    logger.info(f"Time breakdown: pretokenize={pretokenize_time:.2f}s ({pretokenize_time/total_time*100:.1f}%), merge={merge_time:.2f}s ({merge_time/total_time*100:.1f}%)")
    logger.info(f"Final vocab size: {len(vocab)}, Merges performed: {len(merges)}")
    
    return vocab, merges
