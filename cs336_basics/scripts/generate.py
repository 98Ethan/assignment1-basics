"""Text generation script for Transformer language model."""

import torch
import argparse
from cs336_basics.transformer.model import TransformerLM
from cs336_basics.tokenize.tokenizer import Tokenizer


def sample_top_p(logits: torch.Tensor, temperature: float = 1.0, p: float = 0.9) -> int:
    """
    Top-p sampling (or nucleus sampling)
    
    Args:
        logits: 1-D Tensor of logits for each token in the vocabulary
        temperature: Sampling temperature (higher = more random)
        p: Probability threshold for sampling
    Returns:
        int - Sampled token ID

    softmax after masking for numerical stability
    """

    # 1) optional temperature
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-8)

    # 3) sort by logits (desc)
    sorted_probs, sorted_idx = torch.sort(logits, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    # 4) find the cutoff set V(p): the smallest prefix with sum >= p
    cutoff = torch.searchsorted(cumsum, torch.tensor(p, device=probs.device)).item()
    keep = sorted_idx[:cutoff + 1]  # cutoff: first index where cumsum >= p

    # 5) mask & renormalize
    masked_logits = logits.new_full(logits.shape, float('-inf'))
    masked_logits[keep] = logits[keep]    # others are -inf → zero prob
    probs = torch.softmax(masked_logits, dim=-1)
    
    # 6) sample from the masked distribution
    idx = torch.multinomial(probs, 1).item()

    return idx


def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    device: str = "cpu"
) -> str:
    """Generate text using the transformer model.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (higher = more random)
        use_greedy: If True, use greedy sampling instead of random
        device: Device to run on
        
    Returns:
        Generated text
    """
    model.eval()
    
    # Tokenize the prompt
    token_ids = tokenizer.encode(prompt)
    
    # Convert to tensor with batch dimension
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)  # (1, seq_len)
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass - get logits for all positions
            logits = model(input_ids)  # (1, seq_len, vocab_size)
            
            # Get logits for the last position (next token prediction)
            next_token_logits = logits[0, -1, :]  # (vocab_size,)
            
            # Sample next token
            next_token_id = sample_top_p(next_token_logits, temperature)
            
            generated_tokens.append(next_token_id)
            
            # Append to input for next iteration
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
            
            # Stop if hit EOS token (if exists)
            if next_token_id == tokenizer.eos_token_id:
                break
    
    # Decode the generated tokens
    full_tokens = token_ids + generated_tokens
    return tokenizer.decode(full_tokens)


def main():
    parser = argparse.ArgumentParser(description="Generate text using trained transformer")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocab file")
    parser.add_argument("--merges_path", type=str, required=True, help="Path to merges file")
    parser.add_argument("--prompt", type=str, default="The quick brown fox", help="Text prompt")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--p", type=float, default=0.9, help="Top-p sampling threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cpu/cuda)")
    
    # Model hyperparameters (must match the trained model)
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feed-forward dimension")
    parser.add_argument("--theta", type=float, default=10000.0, help="RoPE theta")
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.vocab_path} and {args.merges_path}")
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path)
    
    # Initialize model
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        device=args.device
    )
    
    # Load model weights
    print(f"Loading model weights from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    
    # Generate text
    print(f"\nPrompt: {args.prompt}")
    print("=" * 50)
    
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        use_greedy=args.greedy,
        device=args.device
    )
    
    print(generated_text)


if __name__ == "__main__":
    main()