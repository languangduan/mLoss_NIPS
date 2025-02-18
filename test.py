import torch

# Modified rank_scale function with stable sorting
def rank_scale(row_losses: torch.Tensor) -> torch.Tensor:
    device_ = row_losses.device
    # Create a secondary index to ensure stable sorting
    indices = torch.arange(len(row_losses), device=device_)
    # Sort by values and then by indices for stability
    sorted_indices = torch.argsort(row_losses, stable=True)
    ranks = torch.zeros_like(sorted_indices, dtype=torch.float32, device=device_)
    ranks[sorted_indices] = torch.arange(len(row_losses), dtype=torch.float32, device=device_)
    normalized = ranks / (len(row_losses) - 1 if len(row_losses) > 1 else 1)
    return normalized

# Example input tensor
row_losses = torch.tensor([3.0, 1.0,1.0, 3.0, 2.0])

# Apply the function
normalized_ranks = rank_scale(row_losses)

print(normalized_ranks)
