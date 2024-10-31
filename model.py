# model.py

from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
import copy


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * (torch.log(torch.tensor(10000.0)) / emb_size))
        pos = torch.arange(0, maxlen).unsqueeze(1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        seq_len = token_embedding.size(0)
        pos_embedding = self.pos_embedding[:seq_len, :].unsqueeze(1)
        return self.dropout(token_embedding + pos_embedding)


class EncoderDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(EncoderDecoderLayer, self).__init__()
        # Encoder layer
        self.encoder_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.encoder_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.encoder_norm1 = nn.LayerNorm(d_model)
        self.encoder_norm2 = nn.LayerNorm(d_model)

        # Decoder layer
        self.decoder_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.decoder_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.decoder_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.decoder_norm1 = nn.LayerNorm(d_model)
        self.decoder_norm2 = nn.LayerNorm(d_model)
        self.decoder_norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None
    ):
        # Encoder
        src2 = self.encoder_self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + src2
        src = self.encoder_norm1(src)

        src2 = self.encoder_ffn(src)
        src = src + src2
        src = self.encoder_norm2(src)

        # Decoder
        tgt2 = self.decoder_self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + tgt2
        tgt = self.decoder_norm1(tgt)

        tgt2 = self.decoder_cross_attn(
            tgt, src, src,
            attn_mask=None,
            key_padding_mask=src_key_padding_mask
        )[0]
        tgt = tgt + tgt2
        tgt = self.decoder_norm2(tgt)

        tgt2 = self.decoder_ffn(tgt)
        tgt = tgt + tgt2
        tgt = self.decoder_norm3(tgt)

        return src, tgt


class AlternatingEncoderDecoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        num_layers: int = 6,
        emb_size: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super(AlternatingEncoderDecoder, self).__init__()
        self.embedding = nn.Embedding(num_tokens, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        layer = EncoderDecoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(emb_size)
        self.output_layer = nn.Linear(emb_size, num_tokens)

    def forward(
        self,
        src_input_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None
    ):
        # Embedding and positional encoding
        src_embeddings = self.embedding(src_input_ids)
        src_embeddings = self.positional_encoding(src_embeddings)

        tgt_embeddings = self.embedding(tgt_input_ids)
        tgt_embeddings = self.positional_encoding(tgt_embeddings)

        src = src_embeddings
        tgt = tgt_embeddings

        for layer in self.layers:
            src, tgt = layer(
                src,
                tgt,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

        output = self.norm(tgt)
        logits = self.output_layer(output)
        return logits
