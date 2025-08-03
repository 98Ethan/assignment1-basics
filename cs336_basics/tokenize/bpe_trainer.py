"""Core BPE training implementation using word frequency approach."""

import time
import logging
import heapq
from collections import Counter, defaultdict
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


class WordFrequencyBPE:
    """
    BPE trainer using classic word-frequency approach.
    """
    
    def __init__(self, vocab_size: int, special_tokens: list[str]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        
        # Core data structures
        self.word_splits: Dict[bytes, List[bytes]] = {}  # word -> list of byte pieces
        self.pair_freqs: Dict[tuple[bytes, bytes], int] = {}   # (bytes, bytes) -> frequency
        self.pair_to_words: Dict[tuple[bytes, bytes], Set[bytes]] = defaultdict(set)  # (bytes, bytes) -> set of words containing it
        
        # Heap for efficient max pair selection
        self.max_heap = []
        
        # Results
        self.vocab = {}
        self.merges = []
    
    def initialize_splits_and_pairs(self, word_freqs: Counter):
        """Initialize word splits and pair frequencies."""
        logger.info("Initializing word splits and pair frequencies...")
        start_time = time.time()
        
        for word, freq in word_freqs.items():
            # Split word into individual bytes
            self.word_splits[word] = [bytes([b]) for b in word]
            
            # Count pairs within this word
            word_pieces = self.word_splits[word]

            for pair in zip(word_pieces, word_pieces[1:]): # Pythonic way to iterate consecutive elements
                self.pair_freqs[pair] = self.pair_freqs.get(pair, 0) + freq
                self.pair_to_words[pair].add(word)
        
        # Build initial max heap
        for pair, freq in self.pair_freqs.items():
            heapq.heappush(self.max_heap, (-freq, pair))
        
        elapsed = time.time() - start_time
        logger.info(f"Initialized splits and pairs in {elapsed:.2f}s")
        logger.info(f"Found {len(self.pair_freqs)} unique pairs")
    
    def get_most_frequent_pair(self) -> tuple[bytes, bytes]:
        """Get the most frequent pair using lazy deletion from heap."""
        while self.max_heap:
            neg_freq, pair = heapq.heappop(self.max_heap)
            freq = -neg_freq
            
            # Check if this pair is still valid (lazy deletion)
            if pair in self.pair_freqs and self.pair_freqs[pair] == freq:
                return pair
        
        raise ValueError("No valid pairs found in heap")
    
    def update_pair_frequency(self, pair: tuple[bytes, bytes], freq_delta: int, word: bytes):
        """Update pair frequency and related data structures."""
        if pair in self.pair_freqs:
            self.pair_freqs[pair] += freq_delta
            if self.pair_freqs[pair] <= 0:
                del self.pair_freqs[pair]
                self.pair_to_words[pair].discard(word)
            else:
                # Add updated frequency to heap
                heapq.heappush(self.max_heap, (-self.pair_freqs[pair], pair))
        elif freq_delta > 0:
            # New pair
            self.pair_freqs[pair] = freq_delta
            self.pair_to_words[pair].add(word)
            heapq.heappush(self.max_heap, (-freq_delta, pair))
    
    def apply_merge(self, top_pair: tuple[bytes, bytes], new_token: bytes, word_freqs: Counter):
        """Apply the merge of top_pair -> new_token across all affected words."""
        affected_words = list(self.pair_to_words.get(top_pair, set()))
        
        for word in affected_words:
            word_freq = word_freqs[word]
            word_pieces = self.word_splits[word]
            
            i = 0
            while i < len(word_pieces) - 1:
                if word_pieces[i] == top_pair[0] and word_pieces[i + 1] == top_pair[1]:
                    # Found the pair to merge
                    
                    # Update neighboring pair frequencies before merge
                    if i > 0:
                        # Remove old left pair
                        old_left_pair = (word_pieces[i - 1], word_pieces[i])
                        self.update_pair_frequency(old_left_pair, -word_freq, word)
                        
                        # Add new left pair
                        new_left_pair = (word_pieces[i - 1], new_token)
                        self.update_pair_frequency(new_left_pair, word_freq, word)
                    
                    if i + 2 < len(word_pieces):
                        # Remove old right pair
                        old_right_pair = (word_pieces[i + 1], word_pieces[i + 2])
                        self.update_pair_frequency(old_right_pair, -word_freq, word)
                        
                        # Add new right pair
                        new_right_pair = (new_token, word_pieces[i + 2])
                        self.update_pair_frequency(new_right_pair, word_freq, word)
                    
                    # Perform the actual merge
                    word_pieces[i] = new_token
                    word_pieces.pop(i + 1)
                    
                    # Don't increment i since we removed an element
                else:
                    i += 1
        
        # Remove the merged pair completely
        if top_pair in self.pair_freqs:
            del self.pair_freqs[top_pair]
        if top_pair in self.pair_to_words:
            del self.pair_to_words[top_pair]
    
    def train(self, word_freqs: Counter) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Train BPE using word frequency approach."""
        logger.info("Starting BPE training with word frequency approach...")
        start_time = time.time()
        
        # Initialize vocabulary with byte tokens
        self.vocab = {i: bytes([i]) for i in range(256)}
        next_token_id = 256
        
        # Add special tokens to vocabulary
        for token in self.special_tokens:
            self.vocab[next_token_id] = token.encode('utf-8')
            next_token_id += 1
        
        logger.info(f"Initial vocab size: {len(self.vocab)} (256 bytes + {len(self.special_tokens)} special tokens)")
        
        # Initialize word splits and pair frequencies
        self.initialize_splits_and_pairs(word_freqs)
        
        # Calculate number of merges needed
        num_merges = self.vocab_size - len(self.vocab)
        logger.info(f"Performing {num_merges} BPE merges...")
        
        merge_start = time.time()
        
        for merge_idx in range(num_merges):
            if not self.pair_freqs:
                logger.info(f"No more pairs to merge, stopping at {merge_idx} merges")
                break
            
            # Find most frequent pair
            top_pair = self.get_most_frequent_pair()
            pair_freq = self.pair_freqs[top_pair]
            
            # Create new token
            new_token = top_pair[0] + top_pair[1] # concatenate two bytes objects
            self.vocab[next_token_id] = new_token
            self.merges.append(top_pair)
            
            # Apply merge
            self.apply_merge(top_pair, new_token, word_freqs)
            
            # Log progress
            if (merge_idx + 1) % 50 == 0 or merge_idx < 10:
                elapsed = time.time() - merge_start
                avg_time = elapsed / (merge_idx + 1)
                eta = avg_time * (num_merges - merge_idx - 1)
                try:
                    merged_str = new_token.decode('utf-8', errors='replace')
                    logger.info(f"Merge {merge_idx+1}/{num_merges}: '{merged_str}' (freq={pair_freq}) - {elapsed:.1f}s elapsed, ETA {eta:.1f}s")
                except:
                    logger.info(f"Merge {merge_idx+1}/{num_merges}: {top_pair[0]}+{top_pair[1]} (freq={pair_freq}) - {elapsed:.1f}s elapsed, ETA {eta:.1f}s")
            
            next_token_id += 1
        
        total_time = time.time() - start_time
        logger.info(f"BPE training completed in {total_time:.2f}s")
        logger.info(f"Final vocab size: {len(self.vocab)}, Merges performed: {len(self.merges)}")
        
        return self.vocab, self.merges