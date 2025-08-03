import torch

def test_tensor_memory():
    x = torch.randn(3, 4)  # shape: (rows, cols)
    # x[:, 0]  # A column vector view
    print(x[:, 0].is_contiguous())  # False
    print(x[0, :].is_contiguous())  # True

def test_index():
    values = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    # Suppose we want to select these indices:
    row_indices = torch.tensor([0, 1, 2])
    col_indices = torch.tensor([2, 0, 1])

    row_selected = values[torch.tensor([0, 2])] # selects rows 0 and 2
    print(row_selected) # tensor([[1, 2, 3], [7, 8, 9]])

    selected = values[row_indices, col_indices] # selects values at (0,2), (1,0), (2,1)
    print(selected)  # tensor([3, 4, 8])

    # broadcasting example
    # Step-by-step:
    # 1. token_positions has shape (..., seq_len)
    # 2. Each value in token_positions is used to index the first dimension of self.cos !!!!!!!!
    # 3. For each index i, PyTorch selects self.cos[i, :] (the full d_k slice)
    # 4. The result preserves token_positions' shape (..., seq_len) and appends d_k

    # Example:
    # Concrete example
    cos = torch.randn(1000, 32)  # (max_seq_len, d_k)
    token_positions = torch.randint(0, 200, (10, 3, 50))  # (10, 3, 50), (batch, heads, seq_indices)

    cos_vals = cos[token_positions]  # shape: (10, 3, 50, 32)
    print("cos_vals.shape:", cos_vals.shape)


def test_max():
    x = torch.tensor([
        [[1, 5, 2], [3, 0, 7]],
        [[2, 8, 1], [4, 3, 9]]
    ])

    max_vals, max_indices = torch.max(x, dim=1, keepdim=True)

    print(x.shape)  # (2, 2, 3)
    print(max_vals)    # tensor([[[3, 5, 7]], [[4, 8, 9]]])
    print(max_vals.shape)
    print(x - max_vals)
if __name__ == "__main__":

    # test_tensor_memory()
    # test_index()
    test_max()

