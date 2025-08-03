import regex as re
from multiprocessing import Pool

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def tokenize_text(text: str) -> list[str]:
    """
    Input: str - text to tokenize
    Output: list[str] - list of tokens
    """
    tokens = []
    for match in re.finditer(PAT, text):
        tokens.append(match.group())
    return tokens

def multiprocess_example():
    # Input: list of strings
    text_chunks = [
        "Hello world, this is text to tokenize.",
        "Another chunk with numbers 123 and symbols @#$.",
        "Final chunk with contractions like don't and I'll."
    ]
    
    # Process in parallel using multiprocessing
    with Pool(processes=4) as pool:
        results = pool.map(tokenize_text, text_chunks)
    
    # Output: list of list[str] - each inner list contains tokens for one chunk
    for i, tokens in enumerate(results):
        print(f"Chunk {i+1}: {tokens}")
    
    return results

if __name__ == "__main__":
    multiprocess_example()