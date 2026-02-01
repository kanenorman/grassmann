from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import nn


@dataclass
class LMConfig:
    vocab_size: int
    max_context_len: int
    num_layers: int
    d_model: int = 256
    num_heads: int = 4
    expansion_ratio: int = 4
    dropout_rate: float = 0.1
    weight_tying: bool = True


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_context_len):
        super().__init__()

        if not d_model % num_heads == 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.qkv_projection = nn.Linear(d_model, d_model * 3)
        self.out_projection = nn.Linear(d_model, d_model)
        self.scale = self.d_head**-0.5

        causal_mask = torch.tril(torch.ones(max_context_len, max_context_len))
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x, use_mask=True):
        qkv = self.qkv_projection(x)

        q, k, v = rearrange(
            qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.num_heads
        )

        dots = einsum(q, k, "b h i d, b h j d -> b h i j") * self.scale

        if use_mask:
            seq_len = x.shape[1]
            mask = self.causal_mask[:seq_len, :seq_len].view(1, 1, seq_len, seq_len)
            dots.masked_fill_(mask == 0, float("-inf"))

        attn = F.softmax(dots, dim=-1)

        out = einsum(attn, v, "b h i j, b h j d -> b h i d")

        out = rearrange(out, "b h n d -> b n (h d)")

        return self.out_projection(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, expansion_ratio) -> None:
        super().__init__()
        hidden_dim = d_model * expansion_ratio

        self.layers = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: LMConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadAttention(
            config.d_model, config.num_heads, config.max_context_len
        )
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config.d_model, config.expansion_ratio)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention_norm(x + self.dropout(self.attention(x)))
        x = self.ffn_norm(x + self.dropout(self.ffn(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_context_len, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout_rate)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.weight_tying:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids):
        _, seq_len = token_ids.shape

        token_embeds = self.token_embedding(token_ids)
        position_ids = torch.arange(seq_len, device=token_ids.device)
        position_embeds = self.position_embedding(position_ids)

        hidden_states = token_embeds + position_embeds
        hidden_states = self.embedding_dropout(hidden_states)
        hidden_states = self.transformer_blocks(hidden_states)
        hidden_states = self.output_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits
