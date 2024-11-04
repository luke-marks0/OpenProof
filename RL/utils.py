import torch


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    mask = torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)
    return mask

