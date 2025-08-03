"""
1) Merges-first (slow)
For each (a,b) in merges: scan the whole token list and merge if present.

Work ≈ O(M × N) per word
M = number of merges (often 30k–50k+), N = current token length.

Repeatedly scan for pairs that aren’t there → huge wasted work.

2) Tokens-first (fast)
Look only at adjacent pairs in the current tokens; pick the one with lowest rank and merge; repeat.

First pass builds the set of existing pairs in O(N).

Each iteration:

Find best pair among existing pairs (check rank.get(pair)).

Merge it (ideally all occurrences in one pass).

Only neighbors of the merge change; update candidates locally.

Work ≈ O(R × N) with the simple version (scan pairs each iteration), where R is the number of merges that actually happen for this word (usually small).
With local updates (see below), it’s close to O(N + R).
"""

import regex as re
from typing import Iterable, Iterator


class Tokenizer:
    """BPE Tokenizer for encoding and decoding text."""
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        
        Args:
            vocab: dict[int, bytes] - Mapping from token IDs to their byte representations
            merges: list[tuple[bytes, bytes]] - List of BPE merge rules
            special_tokens: list[str] | None - Optional list of special tokens
        """
        self.vocab = vocab
        self.merges = merges

        self.special_tokens = special_tokens or []
        self.special_token_set = set(self.special_tokens)

        # Sort by length (longest first) for longest-match behavior, refer tests/test_tokenizer.py::test_overlapping_special_tokens
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_token_pattern = f"({'|'.join(re.escape(token) for token in sorted_tokens)})"
        else:
            self.special_token_pattern = None
        
        # Create reverse vocabulary for encoding
        self.vocab_reverse = {v: k for k, v in vocab.items()}
        
        # Build merge dictionary for efficient encoding
        self.merge_dict = {pair: i for i, pair in enumerate(merges)}
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Class method that constructs and returns a Tokenizer from serialized vocabulary and list of merges.
        
        Args:
            vocab_filepath: str - Path to vocabulary file
            merges_filepath: str - Path to merges file
            special_tokens: list[str] | None - Optional list of special tokens
            
        Returns:
            Tokenizer instance
        """
        # Load vocabulary from binary file
        vocab = {}
        with open(vocab_filepath, 'rb') as f:
            vocab_size = int.from_bytes(f.read(4), byteorder='little')
            
            for _ in range(vocab_size):
                token_id = int.from_bytes(f.read(4), byteorder='little')
                token_length = int.from_bytes(f.read(4), byteorder='little')
                token = f.read(token_length)
                vocab[token_id] = token
        
        # Load merges from binary file
        merges = []
        with open(merges_filepath, 'rb') as f:
            num_merges = int.from_bytes(f.read(4), byteorder='little')
            
            for _ in range(num_merges):
                first_length = int.from_bytes(f.read(4), byteorder='little')
                first = f.read(first_length)
                second_length = int.from_bytes(f.read(4), byteorder='little')
                second = f.read(second_length)
                merges.append((first, second))
        
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        
        Args:
            text: str - Input text to encode
            
        Returns:
            list[int] - Sequence of token IDs
        """
        # Handle case with no special tokens
        if not self.special_tokens:
            return self._encode_text_part(text)
        
        # Split text while preserving special tokens
        parts = re.split(self.special_token_pattern, text)
        
        all_token_ids = []
        
        for part in parts:
            # Check if this part is a special token (O(1) lookup)
            if part in self.special_token_set:
                # Special token - add its ID directly
                all_token_ids.append(self.vocab_reverse[part.encode('utf-8')])
            else:
                # Regular text - apply BPE encoding
                if part:  # Skip empty parts
                    token_ids = self._encode_text_part(part)
                    all_token_ids.extend(token_ids)
        
        return all_token_ids
    
    def _apply_merge(self, word: str) -> list[bytes]:
        """
        Apply BPE merges to a single word (core BPE algorithm).
        
        This function implements the core BPE encoding process:
        1. Convert the input word string to UTF-8 bytes
        2. Start with individual bytes as initial tokens
        3. Apply all learned merges in order
        4. Return the final list of BPE tokens
        
        Args:
            word: str - Input word/pretoken as string to apply BPE to
            
        Returns:
            list[bytes] - List of BPE tokens (as bytes) for this word
        """
        # Start with individual bytes for this word
        tokens = [bytes([b]) for b in word.encode("utf-8")]
        
        # Apply merges in order
        for merge_pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and 
                    tokens[i] == merge_pair[0] and 
                    tokens[i + 1] == merge_pair[1]):
                    # Merge the pair
                    new_tokens.append(merge_pair[0] + merge_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens

    def _encode_text_part(self, text: str) -> list[int]:
        """
        Encode a text part (without special tokens) using BPE.
        
        Args:
            text: str - Text part to encode
            
        Returns:
            list[int] - Token IDs for this text part
        """
        # Apply pretokenization pattern to get words
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        words = []
        for match in re.finditer(PAT, text):
            pretoken = match.group()
            words.append(pretoken)
        
        # Apply BPE to each word
        all_tokens = []
        for word in words:
            tokens = self._apply_merge(word)
            all_tokens.extend(tokens)
        
        # Convert tokens to IDs
        token_ids = []
        for token in all_tokens:
            if token in self.vocab_reverse:
                token_ids.append(self.vocab_reverse[token])
            else:
                # Handle unknown tokens - use replacement character
                token_ids.append(65533)  # U+FFFD as integer
        
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings, return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files.
        
        Args:
            iterable: Iterable[str] - Iterable of strings (e.g., file handle)
            
        Yields:
            int - Token IDs one at a time
        """
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        
        Args:
            ids: list[int] - Sequence of token IDs
            
        Returns:
            str - Decoded text
        """
        # Convert IDs to bytes
        byte_tokens = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_tokens.append(self.vocab[token_id])
            else:
                # Handle unknown token IDs - use replacement character
                byte_tokens.append(b'\xef\xbf\xbd')  # Unicode replacement character in UTF-8
        
        # Concatenate all bytes
        all_bytes = b''.join(byte_tokens)
        
        # Decode to string
        try:
            return all_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # 'replace': automatically replace malformed data with the replacement marker.
            return all_bytes.decode('utf-8', errors='replace')