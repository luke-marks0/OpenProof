from typing import Dict, List, Tuple
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from datasets import load_dataset
import re


def load_data() -> List[Dict]:
    """
    Load the dataset and return the training data.

    Returns:
        List[Dict]: The dataset as a list of examples.
    """
    dataset = load_dataset("internlm/Lean-Workbook")
    return dataset['train']


def get_unique_tactics(data: List[Dict]) -> Dict[str, int]:
    """
    Extract unique tactics from the data and assign IDs.

    Args:
        data (List[Dict]): The dataset.

    Returns:
        Dict[str, int]: A mapping from tactic to unique IDs.
    """
    tactics_set = set()
    for example in data:
        tactic = example['tactic'].strip()
        tactics_set.add(tactic)
    tactics_list = sorted(list(tactics_set))
    tactic_to_id = {tactic: idx for idx, tactic in enumerate(tactics_list)}
    return tactic_to_id


def preprocess_data(data: List[Dict], tactic_to_id: Dict[str, int]) -> List[Tuple[str, str, str, int]]:
    """
    Preprocess the data to extract formal_statement, state_before, state_after, and tactic IDs.
    Filters out the prefix "theorem lean_workbook_plus_[number]" and suffix ":= by sorry" from the formal_statement.

    Args:
        data (List[Dict]): The dataset.
        tactic_to_id (Dict[str, int]): Mapping from tactics to IDs.

    Returns:
        List[Tuple[str, str, str, int]]: A list of (formal_statement, state_before, state_after, tactic_id) tuples.
    """
    data_tuples = []
    for example in data:
        formal_statement = example['formal_statement']
        formal_statement = remove_theorem_prefix(formal_statement)
        formal_statement = remove_sorry_suffix(formal_statement)

        state_before = example['state_before']
        state_after = example['state_after']
        tactic = example['tactic'].strip()
        tactic_id = tactic_to_id[tactic]
        data_tuples.append((formal_statement, state_before, state_after, tactic_id))
    return data_tuples


def remove_theorem_prefix(formal_statement: str) -> str:
    """
    Removes the prefix "theorem lean_workbook_plus_[number]" from the formal_statement.

    Args:
        formal_statement (str): The original formal_statement.

    Returns:
        str: The formal_statement without the prefix.
    """
    pattern = r'^theorem lean_workbook_plus_\d+\s*'
    formal_statement = re.sub(pattern, '', formal_statement)
    return formal_statement


def remove_sorry_suffix(formal_statement: str) -> str:
    """
    Removes the suffix ":= by sorry" from the formal_statement.

    Args:
        formal_statement (str): The original formal_statement.

    Returns:
        str: The formal_statement without the suffix.
    """
    pattern = r':=\s*by\s*sorry\s*$'
    formal_statement = re.sub(pattern, '', formal_statement)
    return formal_statement
