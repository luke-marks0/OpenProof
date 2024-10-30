from typing import List, Tuple, Dict
import torch
from torch import nn
from torch.optim import Adam
import wandb
import os
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

from data import load_data, preprocess_data, get_unique_tactics, remove_theorem_prefix, remove_sorry_suffix
from model import Seq2SeqModel


class LeanDataset(Dataset):
    def __init__(
        self,
        data_tuples: List[Tuple[str, str, str, int]],
        tokenizer,
        id_to_tactic: Dict[int, str],
        max_length: int = 512
    ):
        self.data_tuples = data_tuples
        self.tokenizer = tokenizer
        self.id_to_tactic = id_to_tactic
        self.max_length = max_length

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx: int):
        formal_statement, state_before, _, tactic_id = self.data_tuples[idx]
        formal_statement = remove_theorem_prefix(formal_statement)
        formal_statement = remove_sorry_suffix(formal_statement)
        src1_text = state_before
        src2_text = formal_statement
        tactic_text = self.id_to_tactic[tactic_id]
        tactic_text = tactic_text + " [STOP]"
        src1_encoding = self.tokenizer.encode(src1_text)
        src2_encoding = self.tokenizer.encode(src2_text)
        tgt_encoding = self.tokenizer.encode(tactic_text)
        src1_ids = src1_encoding.ids[:self.max_length]
        src2_ids = src2_encoding.ids[:self.max_length]
        tgt_ids = [self.tokenizer.token_to_id("[BOS]")] + tgt_encoding.ids[:self.max_length - 1]
        src1_ids = torch.tensor(src1_ids)
        src2_ids = torch.tensor(src2_ids)
        tgt_ids = torch.tensor(tgt_ids)
        return src1_ids, src2_ids, tgt_ids


def collate_fn(batch):
    src1_ids = [item[0] for item in batch]
    src2_ids = [item[1] for item in batch]
    tgt_ids = [item[2] for item in batch]
    src1_ids = nn.utils.rnn.pad_sequence(src1_ids, padding_value=0)
    src2_ids = nn.utils.rnn.pad_sequence(src2_ids, padding_value=0)
    tgt_ids = nn.utils.rnn.pad_sequence(tgt_ids, padding_value=0)
    return src1_ids, src2_ids, tgt_ids


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generate a square mask for the sequence. Masked positions are filled with float('-inf')."""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def train_model(config: Dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project='OpenProof', config=config)
    config = wandb.config
    data = load_data()
    tactic_to_id = get_unique_tactics(data)
    num_tactics = len(tactic_to_id)
    id_to_tactic = {v: k for k, v in tactic_to_id.items()}
    data_tuples = preprocess_data(data, tactic_to_id)

    if config['train_new_tokenizer']:
        tokenizer = train_tokenizer(data_tuples, id_to_tactic)
        tokenizer_path = 'tokenizer.json'
        tokenizer.save(tokenizer_path)
        tokenizer_artifact = wandb.Artifact('tokenizer', type='tokenizer')
        tokenizer_artifact.add_file(tokenizer_path)
        wandb.log_artifact(tokenizer_artifact)
        os.remove(tokenizer_path)
    else:
        tokenizer = load_tokenizer_from_wandb()

    if "[STOP]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens(["[STOP]"])
        tokenizer_path = 'tokenizer.json'
        tokenizer.save(tokenizer_path)
        tokenizer_artifact = wandb.Artifact('tokenizer', type='tokenizer')
        tokenizer_artifact.add_file(tokenizer_path)
        wandb.log_artifact(tokenizer_artifact)
        os.remove(tokenizer_path)

    vocab_size = tokenizer.get_vocab_size()
    batch_size = config['batch_size']
    dataset = LeanDataset(data_tuples, tokenizer, id_to_tactic)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model_params = config['model']
    model = Seq2SeqModel(
        num_tokens=vocab_size,
        emb_size=model_params['emb_size'],
        nhead=model_params['nhead'],
        num_encoder_layers=model_params['num_encoder_layers'],
        num_decoder_layers=model_params['num_decoder_layers'],
        dim_feedforward=model_params['dim_feedforward'],
        dropout=model_params['dropout']
    )
    model.to(device)
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    num_epochs = config['num_epochs']

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (src1_ids, src2_ids, tgt_ids) in enumerate(dataloader):
            src1_ids = src1_ids.to(device)
            src2_ids = src2_ids.to(device)
            tgt_ids = tgt_ids.to(device)
            optimizer.zero_grad()
            src1_key_padding_mask = (src1_ids == 0).transpose(0, 1)
            src2_key_padding_mask = (src2_ids == 0).transpose(0, 1)
            tgt_key_padding_mask = (tgt_ids == 0).transpose(0, 1)
            tgt_input = tgt_ids[:-1, :]
            tgt_output = tgt_ids[1:, :]
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(device)
            logits = model(
                src1_ids,
                src2_ids,
                tgt_input,
                src1_key_padding_mask=src1_key_padding_mask,
                src2_key_padding_mask=src2_key_padding_mask,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask[:, :-1]
            )
            tgt_output_np = tgt_output.cpu().numpy()
            stop_token_id = tokenizer.token_to_id("[STOP]")
            loss_mask = (tgt_output_np != 0).astype(float)
            for i in range(tgt_output_np.shape[1]):
                stop_indices = (tgt_output_np[:, i] == stop_token_id).nonzero()[0]
                if stop_indices.size > 0:
                    first_stop = stop_indices[0]
                    loss_mask[first_stop + 1:, i] = 0
            loss_mask = torch.tensor(loss_mask, device=device).reshape(-1)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            wandb.log({'train_loss': loss.item(), 'epoch': epoch + 1, 'batch': batch_idx + 1})
            with torch.no_grad():
                predicted_ids = logits.argmax(dim=-1)
                pred_tokens = predicted_ids[:, 0].cpu().numpy()
                tgt_tokens = tgt_output[:, 0].cpu().numpy()
                pred_tokens = pred_tokens[pred_tokens != 0]
                tgt_tokens = tgt_tokens[tgt_tokens != 0]
                if stop_token_id in pred_tokens:
                    stop_index = (pred_tokens == stop_token_id).nonzero()[0][0]
                    pred_tokens = pred_tokens[:stop_index + 1]
                if stop_token_id in tgt_tokens:
                    stop_index = (tgt_tokens == stop_token_id).nonzero()[0][0]
                    tgt_tokens = tgt_tokens[:stop_index + 1]
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=False)
                tgt_text = tokenizer.decode(tgt_tokens, skip_special_tokens=False)
                src1_tokens = src1_ids[:, 0].cpu().numpy()
                src2_tokens = src2_ids[:, 0].cpu().numpy()
                src1_tokens = src1_tokens[src1_tokens != 0]
                src2_tokens = src2_tokens[src2_tokens != 0]
                src1_text = tokenizer.decode(src1_tokens, skip_special_tokens=True)
                src2_text = tokenizer.decode(src2_tokens, skip_special_tokens=True)
                print(f"Epoch: {epoch + 1}, Batch: {batch_idx + 1}")
                print("State Before (src1):", src1_text)
                print("Formal Statement (src2):", src2_text)
                print("Target Tactic:", tgt_text)
                print("Predicted Tactic:", pred_text)
                print("-" * 50)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
    final_model_path = "final_transformer.pt"
    torch.save(model.state_dict(), final_model_path)
    final_model_artifact = wandb.Artifact('final_transformer', type='model')
    final_model_artifact.add_file(final_model_path)
    wandb.log_artifact(final_model_artifact)
    os.remove(final_model_path)
    wandb.finish()


def train_tokenizer(data_tuples: List[Tuple[str, str, str, int]], id_to_tactic: Dict[int, str]):
    from tokenizers.models import WordPiece
    from tokenizers.trainers import WordPieceTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[BOS]", "[STOP]"]
    trainer = WordPieceTrainer(
        vocab_size=30522, special_tokens=special_tokens
    )

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
    wandb.init(project='OpenProof-tokenizer', job_type='download')
    artifact = wandb.use_artifact('tokenizer:latest')
    artifact_dir = artifact.download()
    tokenizer_path = os.path.join(artifact_dir, 'tokenizer.json')
    tokenizer = Tokenizer.from_file(tokenizer_path)
    wandb.finish()
    return tokenizer
