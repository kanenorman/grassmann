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
    tensor_lifting_strategy: str
    lags: list
    d_model: int
    num_heads: int
    expansion_ratio: int
    dropout_rate: float
    weight_tying: bool
    d_low: int
    pre_norm: bool = False


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


class GrassmannMixing(nn.Module):
    def __init__(self, d_model, d_low, lags):
        super().__init__()
        self.d_model = d_model
        self.d_low = d_low
        self.lags = lags

        # 1. Linearly reduce token states to low-dimensional space
        self.reduce = nn.Linear(d_model, d_low)

        # 2. Calculate Plucker embedding dimension (d_low choose 2)
        self.d_plucker = (d_low * (d_low - 1)) // 2

        # 3. Projection layer to map geometric features back to model dimension
        self.expand = nn.Linear(self.d_plucker, d_model)

        # 4. Gating mechanism (paper uses concatenation of x and g)
        self.gate_proj = nn.Linear(2 * d_model, d_model)

        # Precompute indices for plucker coordinates (strictly upper triangular)
        idx_i, idx_j = torch.triu_indices(d_low, d_low, offset=1)
        self.register_buffer("idx_i", idx_i)
        self.register_buffer("idx_j", idx_j)

    def forward(self, x):
        b, n, d = x.shape
        z = self.reduce(x)

        # 1. Accumulate in the LOWER dimensional Plucker space (smaller or similar size)
        plucker_total = torch.zeros(b, n, self.d_plucker, device=x.device)
        counts = torch.zeros(b, n, 1, device=x.device)

        for lag in self.lags:
            if lag <= 0 or lag >= n: continue

            z_curr = z[:, lag:, :]
            z_past = z[:, :-lag, :]

            z_i = z_curr[:, :, self.idx_i]
            z_j = z_curr[:, :, self.idx_j]
            zp_i = z_past[:, :, self.idx_i]
            zp_j = z_past[:, :, self.idx_j]

            plucker = z_i * zp_j - z_j * zp_i

            # Normalization
            plucker_norm = torch.norm(plucker, dim=-1, keepdim=True)
            plucker = plucker / torch.clamp(plucker_norm, min=1e-6)

            # Accumulate RAW plucker features
            plucker_total[:, lag:, :] = plucker_total[:, lag:, :] + plucker
            counts[:, lag:, :] = counts[:, lag:, :] + 1

        # 2. Average in Plucker space
        counts = torch.clamp(counts, min=1.0)
        plucker_avg = plucker_total / counts

        # 3. Project ONCE (Big speedup)
        g_avg = self.expand(plucker_avg)

        # Gated Fusion
        concat = torch.cat([x, g_avg], dim=-1)
        gate = torch.sigmoid(self.gate_proj(concat))
        out = gate * g_avg

        return out


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
        self.pre_norm = config.pre_norm
        self.attention_norm = nn.LayerNorm(config.d_model)

        if config.tensor_lifting_strategy == "grassmann":
            self.attention = GrassmannMixing(
                config.d_model, d_low=config.d_low, lags=config.lags
            )
        elif config.tensor_lifting_strategy == "attention":
            self.attention = MultiHeadAttention(
                config.d_model, config.num_heads, config.max_context_len
            )
        else:
            raise ValueError(
                f"Unknown tensor_lifting_strategy: {config.tensor_lifting_strategy}. "
                f"Must be 'attention' or 'grassmann'"
            )

        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config.d_model, config.expansion_ratio)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            attn_out = self.attention(self.attention_norm(x))
            attn_out = self.dropout(attn_out)
            x = x + attn_out

            ffn_out = self.ffn(self.ffn_norm(x))
            ffn_out = self.dropout(ffn_out)
            x = x + ffn_out
        else:
            attn_out = self.attention(x)
            attn_out = self.dropout(attn_out)
            x = self.attention_norm(x + attn_out)

            ffn_out = self.ffn(x)
            ffn_out = self.dropout(ffn_out)
            x = self.ffn_norm(x + ffn_out)

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
