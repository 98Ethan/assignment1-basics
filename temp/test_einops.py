import torch
from jaxtyping import Float
from einops import einsum, reduce, rearrange

def jaxtyping_basics():
    # Old way:
    # x = torch.ones(2, 2, 1, 3)  # batch seq heads hidden
    x: Float[torch.Tensor, "batch seq heads hidden"] = torch.ones(2, 2, 1, 3)


def einops_einsum():
    # Einsum is generalized matrix multiplication with good bookkeeping.
    x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)
    y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2, 3, 4)
    # Old way:
    # z = x @ y.transpose(-2, -1)  # batch, sequence, sequence
    
    # New (einops) way:
    z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
    
    # 1. Dimensions that are not named in the output are summed over.
    # 2. can use ... to represent broadcasting over any number of dimensions:
    z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")

def einops_reduce():
    # You can reduce a single tensor via some operation (e.g., sum, mean, max, min).
    x: Float[torch.Tensor, "batch seq hidden"] = torch.ones(2, 3, 4)
    # Old way:
    y = x.sum(dim=-1)
    
    # New (einops) way:
    y = reduce(x, "... hidden -> ...", "sum")


def einops_rearrange():
    # Sometimes, a dimension represents two dimensions and you want to operate on one of them.
    
    x: Float[torch.Tensor, "batch seq total_hidden"] = torch.ones(2, 3, 8)
    # where total_hidden is a flattened representation of heads * hidden1 (heads = 2, hidden1 = 4).
    w: Float[torch.Tensor, "hidden1 hidden2"] = torch.ones(4, 4)
    
    # s1. Break up total_hidden into two dimensions (heads and hidden1):
    x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)
    # If the last dimension has values [1,1,1,2,2,2] (length 6), it will be
    # reshaped to [[1,1,1], [2,2,2]] with shape (2, 3) where:
    # - heads=2 creates the first dimension
    # - hidden1=3 becomes the second dimension
    
    # s2. Operate only on hidden1 dim. Perform the transformation by w:
    x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")
    
    # s3. Combine heads and hidden2 back together:
    x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")


def examples():

    ## S1: let's review pytorch broadcasting:

    # 🔹 Rules for broadcasting (right-aligned comparison):
    # If dimensions match or one is 1, broadcasting proceeds.
    # If dimensions don’t match and neither is 1 → ❌ error.

    A = torch.ones(64, 1, 128, 128, 3)
    B = torch.ones(1, 10, 1, 1, 1)

    # - Dim 0: 64 vs 1 → OK → B expands to 64
    # - Dim 1: 1 vs 10 → OK → A expands to 10
    # - Dim 2/3/4: 128 vs 1 → OK → B expands to match

    # A+B or A*B Result: 
    C = (A + B) # (64, 10, 128, 128, 3)
    C = (A * B) # (64, 10, 128, 128, 3)

    # S2: how to use einsum with broadcasting:
    # Use caes: We have a batch of images, and for each image we want to generate 10 dimmed versions based on some scaling factor
    images = torch.randn(64, 128, 128, 3) # (batch, height, width, channel)
    dim_by = torch.linspace(start=0.0, end=1.0, steps=10) # (10,)
    ## Reshape and multiply
    dim_value = rearrange(dim_by,    "dim_value              -> 1 dim_value 1 1 1") # (1, 10, 1, 1, 1)
    images_rearr = rearrange(images, "b height width channel -> b 1 height width channel") # (64, 1, 128, 128, 3)
    dimmed_images = images_rearr * dim_value

    ## Or in one go:
    dimmed_images = einsum(
        images, dim_by,
        "batch height width channel, dim_value -> batch dim_value height width channel"
    )


def test_einsum_patterns():
    """Test if einsum("b c m, m n -> b c n") and einsum("b c m, n m -> b c n") return same thing"""
    b, c, m, n = 2, 3, 4, 5
    x = torch.randn(b, c, m)  # Shape: (2, 3, 4)
    w = torch.randn(m, n)     # Shape: (4, 5)
    
    # Pattern 1: "b c m, m n -> b c n"
    # More standard: - follows the natural matrix multiplication convention (left matrix cols = right matrix rows)
    result1 = einsum(x, w, "b c m, m n -> b c n")
    
    # Pattern 2: "b c m, n m -> b c n" with transposed weight
    result2 = einsum(x, w.T, "b c m, n m -> b c n")
    
    print(f"Results equal: {torch.allclose(result1, result2)}")



if __name__ == "__main__":
    # jaxtyping_basics()
    # einops_einsum()
    # einops_reduce()
    # einops_rearrange()

    # examples()
    test_einsum_patterns()

