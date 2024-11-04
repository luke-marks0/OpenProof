import os
import sys
import argparse
import yaml
import json
import torch
import wandb
import logging
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Any

from tokenizers import Tokenizer
from lean4_environment import Lean4Environment
from policy_value_model import PolicyValueModel

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

logging.basicConfig(level=logging.INFO)


def parse_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[AlternatingEncoderDecoder, Tokenizer, int, Dict[str, Any]]:
    wandb.init(project="OpenProof", job_type="download")

    model_artifact = wandb.use_artifact('final_transformer:latest')
    model_dir = model_artifact.download()
    model_path = os.path.join(model_dir, 'final_transformer.pt')
    model_params_path = os.path.join(model_dir, 'model_params.json')

    with open(model_params_path, 'r') as f:
        model_params = json.load(f)

    num_tokens = model_params['num_tokens']

    base_model = AlternatingEncoderDecoder(
        num_tokens=num_tokens,
        num_layers=model_params["num_layers"],
        emb_size=model_params["emb_size"],
        nhead=model_params["nhead"],
        dim_feedforward=model_params["dim_feedforward"],
        dropout=model_params["dropout"],
        max_seq_length=model_params.get("max_seq_length", 5000)
    )
    base_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    base_model.eval()

    tokenizer_artifact = wandb.use_artifact('tokenizer:latest')
    tokenizer_dir = tokenizer_artifact.download()
    tokenizer_path = os.path.join(tokenizer_dir, 'tokenizer.json')
    tokenizer = Tokenizer.from_file(tokenizer_path)

    wandb.finish()
    return base_model, tokenizer, num_tokens, model_params


class LeanPPODataset(Dataset):
    def __init__(self, data_tuples: List[Tuple[str, str, str, int]]):
        self.data_tuples = data_tuples

    def __len__(self) -> int:
        return len(self.data_tuples)

    def __getitem__(self, idx: int) -> Tuple[str, str, str, int]:
        return self.data_tuples[idx]


def ppo_collate_fn(batch: List[Tuple[str, str, str, int]]) -> Tuple[List[str], List[str], List[str], List[int]]:
    formal_statements, state_befores, state_afters, tactic_ids = zip(*batch)
    return list(formal_statements), list(state_befores), list(state_afters), list(tactic_ids)


def generate_action(
    policy_value_model: PolicyValueModel,
    src_ids: torch.Tensor,
    device: torch.device,
    tokenizer: Tokenizer,
    max_generation_steps: int = 50
) -> Tuple[List[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    policy_value_model.eval()
    tgt_input_ids = torch.tensor([tokenizer.token_to_id("[BOS]")], dtype=torch.long).unsqueeze(1).to(device)
    generated_tactic = []
    action_log_probs = []

    with torch.no_grad():
        tgt_mask = generate_square_subsequent_mask(tgt_input_ids.size(0)).to(device)
        src_key_padding_mask = (src_ids == 0).transpose(0, 1)
        tgt_key_padding_mask = (tgt_input_ids == 0).transpose(0, 1)
        _, value_estimate = policy_value_model(
            src_ids,
            tgt_input_ids,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask
        )
        value_estimate = value_estimate.squeeze(-1)

    for _ in range(max_generation_steps):
        tgt_mask = generate_square_subsequent_mask(tgt_input_ids.size(0)).to(device)
        src_key_padding_mask = (src_ids == 0).transpose(0, 1)
        tgt_key_padding_mask = (tgt_input_ids == 0).transpose(0, 1)
        with torch.no_grad():
            logits, _ = policy_value_model(
                src_ids,
                tgt_input_ids,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_mask=tgt_mask
            )
        logits = logits[-1, :, :]
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action_log_probs.append(log_prob)
        action_id = action.item()
        generated_tactic.append(action_id)
        if action_id == tokenizer.token_to_id("[STOP]"):
            break
        tgt_input_ids = torch.cat([tgt_input_ids, action.unsqueeze(0)], dim=0)
    else:
        logging.warning("Reached maximum generation steps without generating a [STOP] token.")

    total_log_prob = torch.stack(action_log_probs).sum().detach()
    return generated_tactic, total_log_prob, value_estimate, tgt_input_ids.detach()


def ppo_update(
    policy_value_model: PolicyValueModel,
    optimizer: torch.optim.Optimizer,
    old_log_prob: torch.Tensor,
    state: Tuple[torch.Tensor, torch.Tensor],
    action: torch.Tensor,
    return_: torch.Tensor,
    advantage: torch.Tensor,
    clip_param: float = 0.2
) -> Tuple[float, float, float]:
    policy_value_model.train()
    src_input_ids, tgt_input_ids = state
    src_input_ids = src_input_ids.to(return_.device)
    tgt_input_ids = tgt_input_ids.to(return_.device)

    src_key_padding_mask = (src_input_ids == 0).transpose(0, 1)
    tgt_mask = generate_square_subsequent_mask(tgt_input_ids.size(0)).to(src_input_ids.device)
    tgt_key_padding_mask = (tgt_input_ids == 0).transpose(0, 1)

    logits, values = policy_value_model(
        src_input_ids,
        tgt_input_ids,
        src_key_padding_mask=src_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        tgt_mask=tgt_mask
    )

    logits = logits[:-1]
    log_probs = F.log_softmax(logits, dim=-1)
    actions_ids = tgt_input_ids[1:]
    log_probs_actions = log_probs.gather(-1, actions_ids.unsqueeze(-1)).squeeze(-1)
    log_probs_actions_sum = log_probs_actions.sum(dim=0)

    ratio = torch.exp(log_probs_actions_sum - old_log_prob)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(values.squeeze(-1), return_)
    loss = policy_loss + 0.5 * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item()


def train_model(config: Dict[str, Any]) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_model, tokenizer, num_tokens, model_params = load_model_and_tokenizer(config)
    policy_value_model = PolicyValueModel(base_model).to(device)
    optimizer = optim.Adam(policy_value_model.parameters(), lr=config['learning_rate'])

    wandb.init(project="OpenProof-PPO", config=config)

    data = load_data()
    tactic_to_id = get_unique_tactics(data)
    data_tuples = preprocess_data(data, tactic_to_id)

    batch_size = config['batch_size']
    ppo_dataset = LeanPPODataset(data_tuples)
    data_loader = DataLoader(ppo_dataset, batch_size=batch_size, shuffle=True, collate_fn=ppo_collate_fn)

    num_epochs = config['num_epochs']
    clip_param = config['clip_param']

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(data_loader):
            formal_statements, state_befores, _, _ = batch
            formal_statement = remove_sorry_suffix(remove_theorem_prefix(formal_statements[0]))
            env = Lean4Environment(formal_statement, state_befores[0])
            state = env.reset()

            src_text = f"{formal_statement} [EOFS] {state} [EOBS]"
            src_encoding = tokenizer.encode(src_text)
            src_ids = torch.tensor(src_encoding.ids, dtype=torch.long).unsqueeze(1).to(device)

            generated_tactic, total_log_prob, value_estimate, tgt_input_ids = generate_action(
                policy_value_model, src_ids, device, tokenizer, config['max_generation_steps']
            )
            action_text = tokenizer.decode(generated_tactic)
            _, reward, _ = env.step(action_text)
            env.close()

            returns = torch.tensor([reward], dtype=torch.float32).to(device)
            advantages = returns - value_estimate

            loss, policy_loss, value_loss = ppo_update(
                policy_value_model,
                optimizer,
                total_log_prob,
                (src_ids.detach(), tgt_input_ids.detach()),
                tgt_input_ids.detach(),
                returns,
                advantages,
                clip_param=clip_param
            )

            wandb.log({
                "epoch": epoch + 1,
                "batch": batch_idx + 1,
                "total_reward": reward,
                "loss": loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss
            })

    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Post-trains with PPO")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()

    config = parse_config(args.config)
    train_model(config)


if __name__ == "__main__":
    main()
