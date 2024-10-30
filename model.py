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


class Encoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        emb_size: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_tokens, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        embeddings = self.positional_encoding(embeddings)
        memory = self.transformer_encoder(embeddings, src_key_padding_mask=src_key_padding_mask)
        return memory


class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_ff = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2a = nn.LayerNorm(d_model)
        self.norm2b = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2a = nn.Dropout(dropout)
        self.dropout2b = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory1, memory2, tgt_mask=None,
                memory_mask1=None, memory_mask2=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask1=None,
                memory_key_padding_mask2=None):
        tgt2 = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2a = self.cross_attn1(
            tgt, memory1, memory1,
            attn_mask=memory_mask1,
            key_padding_mask=memory_key_padding_mask1
        )[0]
        tgt = tgt + self.dropout2a(tgt2a)
        tgt = self.norm2a(tgt)

        tgt2b = self.cross_attn2(
            tgt, memory2, memory2,
            attn_mask=memory_mask2,
            key_padding_mask=memory_key_padding_mask2
        )[0]
        tgt = tgt + self.dropout2b(tgt2b)
        tgt = self.norm2b(tgt)

        tgt2 = self.linear2(self.dropout_ff(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class CustomTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(CustomTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(decoder_layer.linear2.out_features)

    def forward(self, tgt, memory1, memory2, tgt_mask=None,
                memory_mask1=None, memory_mask2=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask1=None,
                memory_key_padding_mask2=None):
        output = tgt

        for mod in self.layers:
            output = mod(
                output, memory1, memory2,
                tgt_mask=tgt_mask,
                memory_mask1=memory_mask1,
                memory_mask2=memory_mask2,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask1=memory_key_padding_mask1,
                memory_key_padding_mask2=memory_key_padding_mask2
            )

        output = self.norm(output)

        return output


class Decoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        emb_size: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_tokens, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        decoder_layer = CustomTransformerDecoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_decoder = CustomTransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(emb_size, num_tokens)

    def forward(
        self,
        tgt: torch.Tensor,
        memory1: torch.Tensor,
        memory2: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask1: torch.Tensor = None,
        memory_key_padding_mask2: torch.Tensor = None
    ) -> torch.Tensor:
        embeddings = self.embedding(tgt)
        embeddings = self.positional_encoding(embeddings)
        output = self.transformer_decoder(
            embeddings,
            memory1,
            memory2,
            tgt_mask=tgt_mask,
            memory_mask1=None,
            memory_mask2=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask1=memory_key_padding_mask1,
            memory_key_padding_mask2=memory_key_padding_mask2
        )
        logits = self.output_layer(output)
        return logits


class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        emb_size: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super(Seq2SeqModel, self).__init__()
        self.encoder1 = Encoder(
            num_tokens=num_tokens,
            emb_size=emb_size,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.encoder2 = Encoder(
            num_tokens=num_tokens,
            emb_size=emb_size,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.decoder = Decoder(
            num_tokens=num_tokens,
            emb_size=emb_size,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

    def forward(
        self,
        src1: torch.Tensor,
        src2: torch.Tensor,
        tgt: torch.Tensor,
        src1_key_padding_mask: torch.Tensor = None,
        src2_key_padding_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        memory1 = self.encoder1(src1, src_key_padding_mask=src1_key_padding_mask)
        memory2 = self.encoder2(src2, src_key_padding_mask=src2_key_padding_mask)
        logits = self.decoder(
            tgt,
            memory1,
            memory2,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask1=src1_key_padding_mask,
            memory_key_padding_mask2=src2_key_padding_mask
        )
        return logits

