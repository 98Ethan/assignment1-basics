#!/usr/bin/env python3
"""
Comprehensive test of different file handling approaches for large files.
Tests 3 different strategies for processing a 2GB file with multiprocessing.
"""

import os
import time
import psutil
import multiprocessing as mp
from multiprocessing import Pool
from typing import List, Tuple
import regex as re
from collections import Counter

# Test parameters
INPUT_PATH = "./data/TinyStoriesV2-GPT4-train.txt"
NUM_PROCESSES = 4
NUM_CHUNKS = 16  # More chunks than processes
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_BYTES = PAT.encode('utf-8')  # Bytes version for zero-copy processing

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def find_chunk_boundaries(file_path: str, num_chunks: int) -> List[int]:
    """Find chunk boundaries based on special tokens"""
    with open(file_path, "rb") as f:
        # Get total file size
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)
        
        chunk_size = file_size // num_chunks
        boundaries = [i * chunk_size for i in range(num_chunks + 1)]
        boundaries[-1] = file_size
        
        # Adjust boundaries to special tokens
        special_token = b"<|endoftext|>"
        for i in range(1, len(boundaries) - 1):
            f.seek(boundaries[i])
            # Read ahead to find special token
            chunk = f.read(min(8192, file_size - boundaries[i]))
            found_at = chunk.find(special_token)
            if found_at != -1:
                boundaries[i] += found_at
        
        return boundaries

# =============================================================================
# APPROACH 1: Main reads all, workers get full text + chunk info
# =============================================================================

def worker_approach1(args):
    """Worker processes full text but only works on assigned chunk using memoryview - ZERO COPY"""
    full_text_bytes, start_pos, end_pos, chunk_id = args
    
    print(f"Worker {chunk_id}: Processing chunk {start_pos}:{end_pos}")
    start_time = time.time()
    
    # Use memoryview to avoid copying - TRUE zero-copy approach!
    chunk_view = memoryview(full_text_bytes)[start_pos:end_pos]
    
    # Process directly on bytes - no decode/encode needed!
    words = []
    for match in re.finditer(PAT_BYTES, chunk_view):
        # words.append(bytes(match.group()))  # Extract bytes directly
        words.append(match.group())
    
    counter = Counter(words)
    elapsed = time.time() - start_time
    
    print(f"Worker {chunk_id}: Done in {elapsed:.2f}s - {len(counter)} unique words")
    return counter

def test_approach1():
    """Test: Main reads all text, workers get full text + boundaries using TRUE zero-copy"""
    print("\n" + "="*60)
    print("APPROACH 1: Main reads all, workers get memoryview (ZERO-COPY)")
    print("="*60)
    
    start_time = time.time()
    mem_before = get_memory_usage()
    
    # Main process reads entire file as bytes
    print("Main: Reading entire file as bytes...")
    read_start = time.time()
    with open(INPUT_PATH, "rb") as f:
        full_text_bytes = f.read()
    read_time = time.time() - read_start
    
    mem_after_read = get_memory_usage()
    print(f"Main: File read in {read_time:.2f}s")
    print(f"Main: Memory usage: {mem_before:.1f}MB -> {mem_after_read:.1f}MB (+{mem_after_read-mem_before:.1f}MB)")
    
    # Use existing byte boundary function
    boundaries = find_chunk_boundaries(INPUT_PATH, NUM_CHUNKS)
    
    # Prepare worker arguments with bytes and byte boundaries
    worker_args = [
        (full_text_bytes, boundaries[i], boundaries[i+1], i)
        for i in range(len(boundaries) - 1)
    ]
    
    # Process with workers
    print(f"Main: Starting {NUM_PROCESSES} workers for {len(worker_args)} chunks...")
    process_start = time.time()
    
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(worker_approach1, worker_args)
    
    process_time = time.time() - process_start
    
    # Combine results
    total_counter = Counter()
    for counter in results:
        total_counter.update(counter)
    
    total_time = time.time() - start_time
    mem_final = get_memory_usage()
    
    print(f"Results: {len(total_counter)} unique words, {sum(total_counter.values())} total")
    print(f"Timing: Read={read_time:.2f}s, Process={process_time:.2f}s, Total={total_time:.2f}s")
    print(f"Memory: Peak={mem_final:.1f}MB")
    
    return {
        'approach': 'Full text to workers',
        'read_time': read_time,
        'process_time': process_time,
        'total_time': total_time,
        'memory_peak': mem_final,
        'unique_words': len(total_counter),
        'total_words': sum(total_counter.values())
    }

# =============================================================================
# APPROACH 2: Main reads and splits, workers get only their chunk
# =============================================================================

def worker_approach2(args):
    """Worker processes only their assigned text chunk"""
    chunk_text, chunk_id = args
    
    print(f"Worker {chunk_id}: Processing chunk of {len(chunk_text)} chars")
    start_time = time.time()
    
    # Process the chunk
    words = []
    for match in re.finditer(PAT, chunk_text):
        words.append(match.group().encode("utf-8"))
    
    counter = Counter(words)
    elapsed = time.time() - start_time
    
    print(f"Worker {chunk_id}: Done in {elapsed:.2f}s - {len(counter)} unique words")
    return counter

def test_approach2():
    """Test: Main reads and splits text, workers get only their chunk"""
    print("\n" + "="*60)
    print("APPROACH 2: Main reads and splits, workers get only their chunk")
    print("="*60)
    
    start_time = time.time()
    mem_before = get_memory_usage()
    
    # Main process reads and splits file
    print("Main: Reading and splitting file...")
    read_start = time.time()
    with open(INPUT_PATH, "r", encoding="utf-8", errors="ignore") as f:
        full_text = f.read()
    
    # Split text into chunks
    text_len = len(full_text)
    char_boundaries = [i * text_len // NUM_CHUNKS for i in range(NUM_CHUNKS + 1)]
    
    # Adjust boundaries to special tokens
    special_token = "<|endoftext|>"
    for i in range(1, len(char_boundaries) - 1):
        start_search = char_boundaries[i]
        end_search = min(start_search + 1000, text_len)
        chunk = full_text[start_search:end_search]
        found_at = chunk.find(special_token)
        if found_at != -1:
            char_boundaries[i] = start_search + found_at
    
    # Create text chunks
    text_chunks = []
    for i in range(len(char_boundaries) - 1):
        chunk = full_text[char_boundaries[i]:char_boundaries[i+1]]
        text_chunks.append(chunk)
    
    read_time = time.time() - read_start
    mem_after_read = get_memory_usage()
    
    print(f"Main: File read and split in {read_time:.2f}s")
    print(f"Main: Memory usage: {mem_before:.1f}MB -> {mem_after_read:.1f}MB (+{mem_after_read-mem_before:.1f}MB)")
    
    # Prepare worker arguments
    worker_args = [(chunk, i) for i, chunk in enumerate(text_chunks)]
    
    # Process with workers
    print(f"Main: Starting {NUM_PROCESSES} workers for {len(worker_args)} chunks...")
    process_start = time.time()
    
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(worker_approach2, worker_args)
    
    process_time = time.time() - process_start
    
    # Combine results
    total_counter = Counter()
    for counter in results:
        total_counter.update(counter)
    
    total_time = time.time() - start_time
    mem_final = get_memory_usage()
    
    print(f"Results: {len(total_counter)} unique words, {sum(total_counter.values())} total")
    print(f"Timing: Read={read_time:.2f}s, Process={process_time:.2f}s, Total={total_time:.2f}s")
    print(f"Memory: Peak={mem_final:.1f}MB")
    
    return {
        'approach': 'Pre-split chunks to workers',
        'read_time': read_time,
        'process_time': process_time,
        'total_time': total_time,
        'memory_peak': mem_final,
        'unique_words': len(total_counter),
        'total_words': sum(total_counter.values())
    }

# =============================================================================
# APPROACH 3: Each worker reads file independently
# =============================================================================

def worker_approach3(args):
    """Worker reads file independently and processes assigned chunk"""
    file_path, start_byte, end_byte, chunk_id = args
    
    print(f"Worker {chunk_id}: Reading file chunk {start_byte}:{end_byte}")
    start_time = time.time()
    
    # Worker reads only its chunk from file
    with open(file_path, "rb") as f:
        f.seek(start_byte)
        chunk_bytes = f.read(end_byte - start_byte)
        chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
    
    # Process the chunk
    words = []
    for match in re.finditer(PAT, chunk_text):
        words.append(match.group().encode("utf-8"))
    
    counter = Counter(words)
    elapsed = time.time() - start_time
    
    print(f"Worker {chunk_id}: Done in {elapsed:.2f}s - {len(counter)} unique words")
    return counter

def test_approach3():
    """Test: Each worker reads file independently"""
    print("\n" + "="*60)
    print("APPROACH 3: Each worker reads file independently")
    print("="*60)
    
    start_time = time.time()
    mem_before = get_memory_usage()
    
    # Main process only finds boundaries
    print("Main: Finding chunk boundaries...")
    boundaries_start = time.time()
    boundaries = find_chunk_boundaries(INPUT_PATH, NUM_CHUNKS)
    boundaries_time = time.time() - boundaries_start
    
    mem_after_boundaries = get_memory_usage()
    print(f"Main: Boundaries found in {boundaries_time:.2f}s")
    print(f"Main: Memory usage: {mem_before:.1f}MB -> {mem_after_boundaries:.1f}MB")
    
    # Prepare worker arguments with file path and byte boundaries
    worker_args = [
        (INPUT_PATH, boundaries[i], boundaries[i+1], i)
        for i in range(len(boundaries) - 1)
    ]
    
    # Process with workers
    print(f"Main: Starting {NUM_PROCESSES} workers for {len(worker_args)} chunks...")
    process_start = time.time()
    
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(worker_approach3, worker_args)
    
    process_time = time.time() - process_start
    
    # Combine results
    total_counter = Counter()
    for counter in results:
        total_counter.update(counter)
    
    total_time = time.time() - start_time
    mem_final = get_memory_usage()
    
    print(f"Results: {len(total_counter)} unique words, {sum(total_counter.values())} total")
    print(f"Timing: Boundaries={boundaries_time:.2f}s, Process={process_time:.2f}s, Total={total_time:.2f}s")
    print(f"Memory: Peak={mem_final:.1f}MB")
    
    return {
        'approach': 'Independent file reads',
        'read_time': boundaries_time,  # Equivalent to "read" phase
        'process_time': process_time,
        'total_time': total_time,
        'memory_peak': mem_final,
        'unique_words': len(total_counter),
        'total_words': sum(total_counter.values())
    }

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    """Run all tests and compare results"""
    print("File Handling Approach Comparison Test")
    print(f"File: {INPUT_PATH}")
    print(f"File size: {os.path.getsize(INPUT_PATH) / (1024**3):.2f} GB")
    print(f"Processes: {NUM_PROCESSES}")
    print(f"Chunks: {NUM_CHUNKS}")
    
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: File {INPUT_PATH} not found!")
        return
    
    results = []
    
    # Test all approaches
    try:
        result1 = test_approach1()
        results.append(result1)
    except Exception as e:
        print(f"Approach 1 failed: {e}")
    
    try:
        result2 = test_approach2()
        results.append(result2)
    except Exception as e:
        print(f"Approach 2 failed: {e}")
    
    try:
        result3 = test_approach3()
        results.append(result3)
    except Exception as e:
        print(f"Approach 3 failed: {e}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    if not results:
        print("No tests completed successfully!")
        return
    
    print(f"{'Approach':<25} {'Read(s)':<8} {'Process(s)':<10} {'Total(s)':<9} {'Memory(MB)':<11} {'Words'}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['approach']:<25} "
              f"{result['read_time']:<8.2f} "
              f"{result['process_time']:<10.2f} "
              f"{result['total_time']:<9.2f} "
              f"{result['memory_peak']:<11.1f} "
              f"{result['unique_words']:,}")
    
    # Find best approach
    if len(results) > 1:
        fastest_total = min(results, key=lambda x: x['total_time'])
        fastest_process = min(results, key=lambda x: x['process_time'])
        lowest_memory = min(results, key=lambda x: x['memory_peak'])
        
        print(f"\nFastest overall: {fastest_total['approach']} ({fastest_total['total_time']:.2f}s)")
        print(f"Fastest processing: {fastest_process['approach']} ({fastest_process['process_time']:.2f}s)")
        print(f"Lowest memory: {lowest_memory['approach']} ({lowest_memory['memory_peak']:.1f}MB)")
    
    # Verify all approaches got same results
    if len(results) > 1:
        unique_counts = [r['unique_words'] for r in results]
        total_counts = [r['total_words'] for r in results]
        
        if len(set(unique_counts)) == 1 and len(set(total_counts)) == 1:
            print(f"\n✅ All approaches produced identical results: {unique_counts[0]:,} unique words")
        else:
            print(f"\n❌ Results differ! Unique words: {unique_counts}, Total words: {total_counts}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Ensure clean process starts
    main()