import argparse
import yaml
import json
import os
from typing import Dict, Any, List, Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
from tokenizers import Tokenizer

import wandb

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import (
    load_data,
    preprocess_data,
    get_unique_tactics,
    remove_theorem_prefix,
    remove_sorry_suffix,
)
from model import AlternatingEncoderDecoder
from utils import generate_square_subsequent_mask, parse_config


class LeanDataset(Dataset):
    def __init__(
        self,
        data_tuples: List[Tuple[str, str, str, int]],
        tokenizer: Tokenizer,
        id_to_tactic: Dict[int, str],
        max_length: int = 512,
    ):
        self.data_tuples = data_tuples
        self.tokenizer = tokenizer
        self.id_to_tactic = id_to_tactic
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data_tuples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        formal_statement, state_before, _, tactic_id = self.data_tuples[idx]

        formal_statement = remove_theorem_prefix(formal_statement)
        formal_statement = remove_sorry_suffix(formal_statement)

        src_text = formal_statement + " [EOFS] " + state_before + " [EOBS]"
        tgt_text = self.id_to_tactic[tactic_id] + " [STOP]"

        src_encoding = self.tokenizer.encode(src_text)
        tgt_encoding = self.tokenizer.encode(tgt_text)

        src_ids = src_encoding.ids[: self.max_length]
        tgt_ids = [self.tokenizer.token_to_id("[BOS]")] + tgt_encoding.ids[
            : self.max_length - 1
        ]

        src_ids_tensor = torch.tensor(src_ids, dtype=torch.long)
        tgt_ids_tensor = torch.tensor(tgt_ids, dtype=torch.long)

        return src_ids_tensor, tgt_ids_tensor


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    src_ids = [item[0] for item in batch]
    tgt_ids = [item[1] for item in batch]

    src_ids_padded = nn.utils.rnn.pad_sequence(src_ids, padding_value=0)
    tgt_ids_padded = nn.utils.rnn.pad_sequence(tgt_ids, padding_value=0)

    return src_ids_padded, tgt_ids_padded


def prepare_tokenizer(
    data_tuples: List[Tuple[str, str, str, int]],
    id_to_tactic: Dict[int, str],
    config: Dict[str, Any],
) -> Tokenizer:
    if config.get("train_new_tokenizer", False):
        tokenizer = train_tokenizer(data_tuples, id_to_tactic)
        tokenizer_path = "tokenizer.json"
        tokenizer.save(tokenizer_path)
        tokenizer_artifact = wandb.Artifact("tokenizer", type="tokenizer")
        tokenizer_artifact.add_file(tokenizer_path)
        wandb.log_artifact(tokenizer_artifact)
        os.remove(tokenizer_path)
    else:
        tokenizer = load_tokenizer_from_wandb()

    special_tokens = ["[STOP]", "[BOS]", "[EOFS]", "[EOBS]"]
    tokenizer.add_special_tokens(special_tokens)

    tokenizer_path = "tokenizer.json"
    tokenizer.save(tokenizer_path)
    tokenizer_artifact = wandb.Artifact("tokenizer", type="tokenizer")
    tokenizer_artifact.add_file(tokenizer_path)
    wandb.log_artifact(tokenizer_artifact)
    os.remove(tokenizer_path)

    return tokenizer


def create_dataloaders(
    data_tuples: List[Tuple[str, str, str, int]],
    tokenizer: Tokenizer,
    id_to_tactic: Dict[int, str],
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_size = int(0.8 * len(data_tuples))
    val_size = len(data_tuples) - train_size
    train_data, val_data = random_split(data_tuples, [train_size, val_size])

    train_dataset = LeanDataset(train_data, tokenizer, id_to_tactic)
    val_dataset = LeanDataset(val_data, tokenizer, id_to_tactic)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tokenizer: Tokenizer,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (src_ids, tgt_ids) in enumerate(dataloader):
        src_ids = src_ids.to(device)
        tgt_ids = tgt_ids.to(device)

        optimizer.zero_grad()

        src_key_padding_mask = (src_ids == 0).transpose(0, 1)
        tgt_key_padding_mask = (tgt_ids == 0).transpose(0, 1)

        tgt_input = tgt_ids[:-1, :]
        tgt_output = tgt_ids[1:, :]

        tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]

        tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(device)

        logits, _ = model(
            src_input_ids=src_ids,
            tgt_input_ids=tgt_input,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask,
        )

        loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        wandb.log({"train_loss": loss.item(), "epoch": epoch + 1, "batch": batch_idx + 1})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tokenizer: Tokenizer,
    epoch: int,
) -> float:
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch_idx, (src_ids, tgt_ids) in enumerate(dataloader):
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)

            src_key_padding_mask = (src_ids == 0).transpose(0, 1)
            tgt_key_padding_mask = (tgt_ids == 0).transpose(0, 1)

            tgt_input = tgt_ids[:-1, :]
            tgt_output = tgt_ids[1:, :]

            tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]

            tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(device)

            logits, _ = model(
                src_input_ids=src_ids,
                tgt_input_ids=tgt_input,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_mask=tgt_mask,
            )

            loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(dataloader)
    return avg_val_loss


def train_model(config: Dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="OpenProof", config=config)
    config = wandb.config

    data = load_data()
    tactic_to_id = get_unique_tactics(data)
    id_to_tactic = {v: k for k, v in tactic_to_id.items()}
    data_tuples = preprocess_data(data, tactic_to_id)

    tokenizer = prepare_tokenizer(data_tuples, id_to_tactic, config)
    vocab_size = tokenizer.get_vocab_size()

    batch_size = config["batch_size"]
    train_dataloader, val_dataloader = create_dataloaders(
        data_tuples, tokenizer, id_to_tactic, batch_size
    )

    model_params = config["model"]
    model = AlternatingEncoderDecoder(
        num_tokens=vocab_size,
        num_layers=model_params["num_layers"],
        emb_size=model_params["emb_size"],
        nhead=model_params["nhead"],
        dim_feedforward=model_params["dim_feedforward"],
        dropout=model_params["dropout"],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    num_epochs = config["num_epochs"]
    for epoch in range(num_epochs):
        avg_train_loss = train_epoch(
            model, train_dataloader, criterion, optimizer, device, tokenizer, epoch
        )
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg Training Loss: {avg_train_loss:.4f}")

        avg_val_loss = validate_epoch(
            model, val_dataloader, criterion, device, tokenizer, epoch
        )
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg Validation Loss: {avg_val_loss:.4f}")

        wandb.log({"val_loss": avg_val_loss, "epoch": epoch + 1})

    final_model_path = "final_transformer.pt"
    torch.save(model.state_dict(), final_model_path)

    model_params_path = 'model_params.json'
    model_params_to_save = model_params.copy()
    model_params_to_save['num_tokens'] = vocab_size
    with open(model_params_path, 'w') as f:
        json.dump(model_params_to_save, f)

    final_model_artifact = wandb.Artifact("final_transformer", type="model")
    final_model_artifact.add_file(final_model_path)
    final_model_artifact.add_file(model_params_path)
    wandb.log_artifact(final_model_artifact)

    os.remove(final_model_path)
    os.remove(model_params_path)

    wandb.finish()


def train_tokenizer(
    data_tuples: List[Tuple[str, str, str, int]], id_to_tactic: Dict[int, str]
) -> Tokenizer:
    from tokenizers.models import WordPiece
    from tokenizers.trainers import WordPieceTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    special_tokens = [
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[PAD]",
        "[MASK]",
        "[BOS]",
        "[STOP]",
        "[EOFS]",
        "[EOBS]",
    ]
    trainer = WordPieceTrainer(vocab_size=30522, special_tokens=special_tokens)

    texts = []
    for formal_statement, state_before, state_after, tactic_id in data_tuples:
        formal_statement = remove_theorem_prefix(formal_statement)
        formal_statement = remove_sorry_suffix(formal_statement)
        texts.extend([formal_statement, state_before, state_after])
        tactic_text = id_to_tactic[tactic_id] + " [STOP]"
        texts.append(tactic_text)

    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


def load_tokenizer_from_wandb() -> Tokenizer:
    wandb.init(project="OpenProof-tokenizer", job_type="download")
    artifact = wandb.use_artifact("tokenizer:latest")
    artifact_dir = artifact.download()
    tokenizer_path = os.path.join(artifact_dir, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    wandb.finish()
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Trains the prover")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()

    config = parse_config(args.config)
    train_model(config)


if __name__ == '__main__':
    main()
