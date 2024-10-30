import argparse
import yaml
from train import train_model
from typing import Dict


def parse_config(config_path: str) -> Dict:
    """
    Parse the YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        Dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Trains the prover")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()

    config = parse_config(args.config)
    train_model(config)


if __name__ == '__main__':
    main()

