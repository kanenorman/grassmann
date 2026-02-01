from typing import Annotated

import torch
from einops import rearrange
from pydantic import BaseModel, Field
from torch import nn


class ModelConfig(BaseModel):
    vocab_size: Annotated[
        int, Field(gt=0, description="Number of tokens in the vocabulary")
    ]
    max_context_len: Annotated[
        int, Field(gt=0, description="Maximum sequence length the model can process")
    ]
    d_model: Annotated[
        int, Field(gt=0, description="Size of the token embedding vector")
    ]
    num_heads: Annotated[
        int,
        Field(gt=0, description="Number of attention heads in each transformer block"),
    ]
    num_layers: Annotated[
        int, Field(gt=0, description="Number of transformer layers (blocks)")
    ]
    dropout_rate: Annotated[
        float, Field(ge=0.0, le=1.0, description="Dropout rate for regularization")
    ] = 0.0
    use_qkv_bias: Annotated[
        bool, Field(description="Whether to include bias terms in QKV projections")
    ] = False


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.scale = self.head_dim**-0.5

        self.query_proj = nn.Linear(
            config.d_model, config.d_model, bias=config.use_qkv_bias
        )
        self.key_proj = nn.Linear(
            config.d_model, config.d_model, bias=config.use_qkv_bias
        )
        self.value_proj = nn.Linear(
            config.d_model, config.d_model, bias=config.use_qkv_bias
        )
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(config.max_context_len, config.max_context_len), diagonal=1
            ).bool(),
        )

    def forward(self, x, use_causal_mask: bool = True):
        batch_size, seq_len, embed_dim = x.shape

        queries = self.query_proj(x)
        keys = self.key_proj(x)
        values = self.value_proj(x)

        queries = rearrange(queries, "b t (h d) -> b h t d", h=self.num_heads)
        keys = rearrange(keys, "b t (h d) -> b h t d", h=self.num_heads)
        values = rearrange(values, "b t (h d) -> b h t d", h=self.num_heads)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        if use_causal_mask:
            attention_scores.masked_fill_(
                self.causal_mask[:seq_len, :seq_len], -torch.inf
            )

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, values)
        attention_output = rearrange(attention_output, "b h t d -> b t (h d)")
        attention_output = self.out_proj(attention_output)

        return attention_output


class FeedForward(nn.Module):
    EXPANSION_RATIO: int = 4

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_dim = config.d_model * self.EXPANSION_RATIO
        self.layers = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadAttention(config)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention_norm(x + self.dropout(self.attention(x)))
        x = self.ffn_norm(x + self.dropout(self.ffn(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_context_len, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout_rate)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape

        token_embeds = self.token_embedding(token_ids)
        position_ids = torch.arange(seq_len, device=token_ids.device)
        position_embeds = self.position_embedding(position_ids)

        hidden_states = token_embeds + position_embeds
        hidden_states = self.embedding_dropout(hidden_states)
        hidden_states = self.transformer_blocks(hidden_states)
        hidden_states = self.output_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits
