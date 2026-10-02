"""MOTANAXY Model — Byte-level causal Transformer for code generation.

v2 architecture improvements over v1:
- RMSNorm (faster than LayerNorm, used by LLaMA/Gemma)
- RoPE — Rotary Position Embeddings (better length generalization)
- SwiGLU feedforward (better gradient flow, used by LLaMA/Mistral)
- Dropout (attention, residual, embedding) to combat overfitting
- KV-cache for O(n) inference instead of O(n²)
- Backward compatible: old v1 checkpoints still load correctly
"""

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Model Presets
# ---------------------------------------------------------------------------

PRESETS = {
    # v1 presets (backward compatible, LayerNorm + learned pos + GELU MLP)
    "nano":   dict(width=64,  context=128,  layers=2,  heads=4),
    "micro":  dict(width=128, context=256,  layers=4,  heads=4),
    # v2 presets (RMSNorm + RoPE + SwiGLU + Dropout)
    "small":  dict(width=256, context=512,  layers=6,  heads=8,  version=2),
    "medium": dict(width=512, context=1024, layers=8,  heads=8,  version=2),
    "large":  dict(width=192, context=256,  layers=8,  heads=6,  version=2, dropout=0.1),
    "xlarge": dict(width=256, context=256,  layers=12, heads=8,  version=2, dropout=0.1, vocab_size=4096),
}


# ---------------------------------------------------------------------------
# v2 Components
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster than LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def _precompute_rope(dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE frequency tensor as complex exponentials."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)  # (max_len, dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, rope: torch.Tensor, offset: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    T = q.shape[2]
    rope = rope[offset : offset + T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, dim//2)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        # Reshape last dim into pairs, view as complex, rotate, view as real
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * rope
        return torch.view_as_real(x_rotated).flatten(-2).type_as(x)

    return rotate(q), rotate(k)


class SwiGLU(nn.Module):
    """SwiGLU feedforward network (used by LLaMA, Mistral, Gemma).

    SwiGLU(x) = (SiLU(W_gate · x) ⊙ W_up · x) · W_down
    Hidden dim is 8/3 * width, rounded to nearest multiple of 64 for efficiency.
    """

    def __init__(self, width: int, dropout: float = 0.0):
        super().__init__()
        # 8/3 ratio is the standard for SwiGLU to match GELU MLP parameter count
        hidden = int(8 * width / 3)
        # Round to nearest multiple of 64 for hardware efficiency
        hidden = ((hidden + 63) // 64) * 64
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class CausalSelfAttentionV2(nn.Module):
    """Multi-head causal self-attention with RoPE and optional KV-cache."""

    def __init__(self, width: int, heads: int, context: int, dropout: float = 0.0):
        super().__init__()
        assert width % heads == 0
        self.heads = heads
        self.head_dim = width // heads
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.out_proj = nn.Linear(width, width, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Pre-register causal mask
        self.register_buffer(
            "mask",
            torch.ones(context, context, dtype=torch.bool).triu(1),
        )
        # Pre-compute RoPE frequencies
        self.register_buffer(
            "rope",
            _precompute_rope(self.head_dim, context),
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        offset = 0
        if kv_cache is not None:
            offset = kv_cache[0].shape[2]
        q, k = _apply_rope(q, k, self.rope, offset)

        # Append to KV-cache if provided
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)

        # Scaled dot-product attention
        S = k.shape[2]  # total sequence length with cache
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # Apply causal mask (only for non-cached tokens)
        if kv_cache is None:
            att = att.masked_fill(self.mask[:T, :T], float("-inf"))
        else:
            # During cached generation, new tokens can attend to all cached + self
            # No mask needed for single-token generation
            pass

        att = self.attn_dropout(att.softmax(dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out)), new_cache


class TransformerBlockV2(nn.Module):
    """Pre-norm Transformer block with RMSNorm + SwiGLU."""

    def __init__(self, width: int, heads: int, context: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attn = CausalSelfAttentionV2(width, heads, context, dropout)
        self.norm2 = RMSNorm(width)
        self.ffn = SwiGLU(width, dropout)

    def forward(
        self, x: torch.Tensor, kv_cache: Optional[tuple] = None
    ) -> tuple[torch.Tensor, tuple]:
        residual = x
        out, new_cache = self.attn(self.norm1(x), kv_cache)
        x = residual + out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


# ---------------------------------------------------------------------------
# v1 Components (kept for backward compatibility with old checkpoints)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with pre-computed mask."""

    def __init__(self, width: int, heads: int, context: int):
        super().__init__()
        assert width % heads == 0
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        # Pre-register causal mask (Ponytail: compute once, reuse forever)
        self.register_buffer(
            "mask",
            torch.ones(context, context, dtype=torch.bool).triu(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product with causal mask
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:T, :T], float("-inf"))
        att = att.softmax(dim=-1)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block (more stable training)."""

    def __init__(self, width: int, heads: int, context: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = CausalSelfAttention(width, heads, context)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Unified Model
# ---------------------------------------------------------------------------

class MotanaxyModel(nn.Module):
    """Byte-level causal language model for code generation.

    Supports two architecture versions:
    - v1: LayerNorm + learned positions + GELU MLP (original)
    - v2: RMSNorm + RoPE + SwiGLU + Dropout + KV-cache (new)

    Version is auto-detected from config. Old checkpoints load seamlessly.
    """

    def __init__(self, width=64, context=128, layers=2, heads=4,
                 version=1, dropout=0.0, vocab_size=256):
        super().__init__()
        self.config = dict(
            width=width, context=context, layers=layers, heads=heads,
            version=version, dropout=dropout, vocab_size=vocab_size,
        )
        self.version = version

        # Token embedding (shared between v1 and v2)
        self.token_emb = nn.Embedding(vocab_size, width)

        if version == 1:
            # v1: learned positional embeddings
            self.pos_emb = nn.Embedding(context, width)
            self.blocks = nn.ModuleList([
                TransformerBlock(width, heads, context)
                for _ in range(layers)
            ])
            self.ln_final = nn.LayerNorm(width)
        else:
            # v2: RoPE (no pos_emb needed), RMSNorm, SwiGLU, Dropout
            self.emb_dropout = nn.Dropout(dropout)
            self.blocks = nn.ModuleList([
                TransformerBlockV2(width, heads, context, dropout)
                for _ in range(layers)
            ])
            self.ln_final = RMSNorm(width)

        self.head = nn.Linear(width, vocab_size, bias=False)

        # Weight tying: share token embedding and output head
        # (Ponytail: reuse weights, 30% fewer parameters)
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        """GPT-style weight initialization with depth scaling for residuals."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scale residual projections by 1/sqrt(2*layers) for deep networks
        if self.version == 2:
            scale = (2 * self.config["layers"]) ** -0.5
            for block in self.blocks:
                nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=0.02 * scale)
                nn.init.normal_(block.ffn.down.weight, mean=0.0, std=0.02 * scale)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        x = self.token_emb(x)

        if self.version == 1:
            pos = torch.arange(T, device=x.device)
            x = x + self.pos_emb(pos)
            for block in self.blocks:
                x = block(x)
        else:
            x = self.emb_dropout(x)
            for block in self.blocks:
                x, _ = block(x, kv_cache=None)

        x = self.ln_final(x)
        return self.head(x)

    @torch.no_grad()
    def generate(
        self,
        prompt_bytes: list[int],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 40,
    ) -> list[int]:
        """Auto-regressive generation with top-k sampling.

        Uses KV-cache for v2 models (O(n) instead of O(n²)).
        """
        self.eval()
        ids = list(prompt_bytes)
        ctx = self.config["context"]
        dev = self.token_emb.weight.device

        if self.version == 1:
            # v1: recompute full context each step (original behavior)
            for _ in range(max_tokens):
                window = ids[-ctx:]
                x = torch.tensor([window], device=dev)
                logits = self(x)[0, -1] / temperature
                if top_k > 0:
                    top_vals, _ = logits.topk(min(top_k, logits.size(-1)))
                    logits[logits < top_vals[-1]] = float("-inf")
                probs = logits.softmax(-1)
                ids.append(torch.multinomial(probs, 1).item())
        else:
            # v2: KV-cache generation
            # First pass: encode full prompt
            window = ids[-ctx:]
            x = torch.tensor([window], device=dev)
            tok = self.token_emb(x)
            tok = self.emb_dropout(tok)

            caches = [None] * len(self.blocks)
            for i, block in enumerate(self.blocks):
                tok, caches[i] = block(tok, kv_cache=None)

            tok = self.ln_final(tok)
            logits = self.head(tok)[0, -1] / temperature
            if top_k > 0:
                top_vals, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < top_vals[-1]] = float("-inf")
            probs = logits.softmax(-1)
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id)

            # Subsequent steps: single-token with KV-cache
            for _ in range(max_tokens - 1):
                cache_len = caches[0][0].shape[2]
                if cache_len >= ctx:
                    # Cache full: trim oldest tokens
                    trim = cache_len - ctx + 1
                    caches = [
                        (k[:, :, trim:, :], v[:, :, trim:, :])
                        for k, v in caches
                    ]

                x = torch.tensor([[ids[-1]]], device=dev)
                tok = self.token_emb(x)

                for i, block in enumerate(self.blocks):
                    tok, caches[i] = block(tok, kv_cache=caches[i])

                tok = self.ln_final(tok)
                logits = self.head(tok)[0, -1] / temperature
                if top_k > 0:
                    top_vals, _ = logits.topk(min(top_k, logits.size(-1)))
                    logits[logits < top_vals[-1]] = float("-inf")
                probs = logits.softmax(-1)
                ids.append(torch.multinomial(probs, 1).item())

        return ids


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def from_preset(name: str) -> MotanaxyModel:
    """Create a model from a named preset."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Choose from: {list(PRESETS)}")
    return MotanaxyModel(**PRESETS[name])


def load_checkpoint(path: str, device: str = "cpu") -> MotanaxyModel:
    """Load a saved MOTANAXY checkpoint.

    Backward compatible: old v1 checkpoints without 'version' or 'dropout'
    in their config are loaded as version=1, dropout=0.0.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    config = dict(ckpt["config"])
    # Backward compatibility: old checkpoints lack v2 fields
    config.setdefault("version", 1)
    config.setdefault("dropout", 0.0)
    model = MotanaxyModel(**config)
    model.load_state_dict(ckpt["state"])
    return model
