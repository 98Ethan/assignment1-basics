#!/usr/bin/env python3

def read_sample(filepath, num_bytes=2048, output_file="sample_output.txt"):
    """Read a fixed sample from the beginning of a large file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read(num_bytes)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"First {num_bytes} bytes from {filepath} saved to {output_file}")
    print(f"Sample content:")
    print("-" * 50)
    print(content[:500] + "..." if len(content) > 500 else content)
    print("-" * 50)

if __name__ == "__main__":
    filepath = "data/owt_train.txt"
    read_sample(filepath, 20480)