import torch
import yaml
from typing import Dict


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    mask = torch.triu(torch.ones((sz, sz), dtype=torch.bool), diagonal=1)
    return mask


def parse_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
