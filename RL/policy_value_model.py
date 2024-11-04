import torch
import torch.nn as nn


class PolicyValueModel(nn.Module):
    def __init__(self, base_model):
        super(PolicyValueModel, self).__init__()
        self.base_model = base_model
        self.emb_size = base_model.emb_size
        self.value_head = nn.Linear(self.emb_size, 1)

    def forward(self, src_input_ids, tgt_input_ids,
                src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
        logits, decoder_output = self.base_model(
            src_input_ids,
            tgt_input_ids,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask
        )

        value = self.value_head(decoder_output.mean(dim=0))
        return logits, value
