"""
Transformer building blocks, written from scratch.

Every attention variant in this repo is the same operation, softmax(QK^T / sqrt(d)) V,
with a different mask or a different source for K and V. So there is ONE Attention
class and the variants are flags:

    variant                 how to get it
    ----------------------  ---------------------------------------------
    encoder self-attention  causal=False                (sees the whole sequence)
    decoder self-attention  causal=True                 (sees only the past)
    cross-attention         forward(x, memory=enc_out)  (Q from decoder, K/V from encoder)
    padding mask            forward(x, key_padding_mask=...)
    multi-head (MHA)        n_kv_heads = n_heads        (default)
    grouped-query (GQA)     1 < n_kv_heads < n_heads    (Llama 2/3, Mistral)
    multi-query (MQA)       n_kv_heads = 1              (PaLM, Falcon)
    sliding-window          window=w                    (Mistral, Longformer-style local)
    KV cache                forward(x, cache={})        (fast autoregressive decoding)

Models: MiniGPT (decoder-only), BERT (encoder-only), Seq2Seq (encoder-decoder).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads=None, causal=False, window=None, dropout=0.0):
        super().__init__()
        n_kv_heads = n_kv_heads or n_heads
        assert dim % n_heads == 0, "dim must split evenly across heads"
        assert n_heads % n_kv_heads == 0, "query heads must split evenly across K/V heads"

        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = dim // n_heads
        self.causal, self.window = causal, window

        self.q = nn.Linear(dim, dim)
        # GQA/MQA saving: fewer K/V heads -> smaller projection and a smaller KV cache
        self.kv = nn.Linear(dim, 2 * n_kv_heads * self.head_dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.attn_weights = None  # last softmax map, kept in eval mode for heatmaps

    def forward(self, x, memory=None, key_padding_mask=None, cache=None):
        """
        x:                (B, T, D)  queries come from here
        memory:           (B, S, D)  if given -> cross-attention, K/V come from here
        key_padding_mask: (B, S)     True where the key is padding (never attended to)
        cache:            dict       self-attention KV cache, filled in place
        """
        B, T, D = x.shape
        src = x if memory is None else memory

        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)        # (B, H, T, hd)
        k, v = (self.kv(src)
                .view(B, src.size(1), 2, self.n_kv_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4))                                            # 2 x (B, Hkv, S, hd)

        if cache is not None and memory is None:
            # ponytail: cache grows without bound; trim to `window` if memory matters
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        # GQA / MQA: each K/V head is shared by n_heads / n_kv_heads query heads
        group = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(group, dim=1)                                        # (B, H, S, hd)
        v = v.repeat_interleave(group, dim=1)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)                   # (B, H, T, S)

        allowed = self.build_mask(T, k.size(2), key_padding_mask, x.device)
        if allowed is not None:
            att = att.masked_fill(~allowed, float("-inf"))

        att = F.softmax(att, dim=-1)
        if not self.training:
            self.attn_weights = att.detach()
        att = self.attn_drop(att)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, D)                   # merge heads
        return self.resid_drop(self.proj(out))

    def build_mask(self, T, S, key_padding_mask, device):
        """Boolean mask, True = query may attend to key. Shape broadcasts to (B, H, T, S)."""
        allowed = None
        if self.causal or self.window:
            # with a KV cache the T new queries sit at the END of the S keys
            q_pos = torch.arange(S - T, S, device=device)[:, None]
            k_pos = torch.arange(S, device=device)[None, :]
            allowed = torch.ones(T, S, dtype=torch.bool, device=device)
            if self.causal:
                allowed &= k_pos <= q_pos                    # no peeking at the future
            if self.window:
                allowed &= (q_pos - k_pos).abs() < self.window  # only the last `window` tokens
        if key_padding_mask is not None:
            pad_ok = ~key_padding_mask[:, None, None, :]      # (B, 1, 1, S)
            allowed = pad_ok if allowed is None else allowed & pad_ok
        return allowed


class Block(nn.Module):
    """
    Pre-LayerNorm Transformer block.
      cross=False -> encoder block (causal=False) or GPT block (causal=True)
      cross=True  -> seq2seq decoder block: masked self-attn, then cross-attn to the encoder
    """

    def __init__(self, dim, n_heads, causal, cross=False, n_kv_heads=None, window=None, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.self_attn = Attention(dim, n_heads, n_kv_heads, causal, window, dropout)
        self.ln2 = nn.LayerNorm(dim) if cross else None
        self.cross_attn = Attention(dim, n_heads, dropout=dropout) if cross else None
        self.ln3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, memory=None, pad_mask=None, memory_pad_mask=None, cache=None):
        x = x + self.self_attn(self.ln1(x), key_padding_mask=pad_mask, cache=cache)
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.ln2(x), memory=memory, key_padding_mask=memory_pad_mask)
        return x + self.mlp(self.ln3(x))


class MiniGPT(nn.Module):
    """Decoder-only language model (GPT)."""

    def __init__(self, vocab_size, context_length, dim, n_blocks, n_heads,
                 n_kv_heads=None, window=None, dropout=0.1):
        super().__init__()
        self.context_length = context_length
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(context_length, dim)
        self.blocks = nn.ModuleList(
            Block(dim, n_heads, causal=True, n_kv_heads=n_kv_heads, window=window, dropout=dropout)
            for _ in range(n_blocks)
        )
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx, caches=None, start_pos=0):
        T = idx.size(1)
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for i, block in enumerate(self.blocks):
            x = block(x, cache=None if caches is None else caches[i])
        return self.head(self.ln_f(x))                                               # (B, T, vocab)

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=1.0, top_k=None, use_cache=True):
        self.eval()
        caches, pos = None, 0
        for _ in range(n_new):
            if not use_cache:
                logits = self(idx[:, -self.context_length:])        # recompute everything, O(T^2) per token
            elif caches is None or pos >= self.context_length:
                # (re)fill the cache with a prefix pass.
                # ponytail: learned absolute positions can't slide, so a full cache is rebuilt
                # from the last half of the context; RoPE would allow a rolling cache instead
                keep = self.context_length if caches is None else self.context_length // 2
                ctx = idx[:, -keep:]
                caches = [{} for _ in self.blocks]
                logits = self(ctx, caches)
                pos = ctx.size(1)
            else:
                logits = self(idx[:, -1:], caches, start_pos=pos)   # only the newest token, O(T)
                pos += 1

            logits = logits[:, -1] / temperature
            if top_k:
                kth = torch.topk(logits, top_k).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


class BERT(nn.Module):
    """Encoder-only model trained by masked language modelling: predict hidden tokens from BOTH sides."""

    def __init__(self, vocab_size, context_length, dim, n_blocks, n_heads, pad_id, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(context_length, dim)
        self.blocks = nn.ModuleList(Block(dim, n_heads, causal=False, dropout=dropout) for _ in range(n_blocks))
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx):
        pad = idx == self.pad_id
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(idx.size(1), device=idx.device))
        for block in self.blocks:
            x = block(x, pad_mask=pad)          # bidirectional: every token sees every non-pad token
        return self.head(self.ln_f(x))                                               # (B, T, vocab)


class Seq2Seq(nn.Module):
    """Encoder-decoder Transformer ("Attention Is All You Need")."""

    def __init__(self, vocab_size, max_len, dim, n_blocks, n_heads, pad_id, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, dim)  # one vocab shared by source and target
        self.pos_emb = nn.Embedding(max_len, dim)
        self.encoder = nn.ModuleList(Block(dim, n_heads, causal=False, dropout=dropout) for _ in range(n_blocks))
        self.decoder = nn.ModuleList(Block(dim, n_heads, causal=True, cross=True, dropout=dropout)
                                     for _ in range(n_blocks))
        self.ln_enc = nn.LayerNorm(dim)
        self.ln_dec = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def embed(self, idx):
        return self.tok_emb(idx) + self.pos_emb(torch.arange(idx.size(1), device=idx.device))

    def encode(self, src):
        pad = src == self.pad_id
        x = self.embed(src)
        for block in self.encoder:
            x = block(x, pad_mask=pad)          # bidirectional, but never attends to padding
        return self.ln_enc(x), pad

    def decode(self, tgt, memory, memory_pad):
        # target padding sits at the right end, so the causal mask already keeps real tokens
        # from seeing it; padded positions are dropped from the loss with ignore_index
        x = self.embed(tgt)
        for block in self.decoder:
            x = block(x, memory=memory, memory_pad_mask=memory_pad)
        return self.head(self.ln_dec(x))

    def forward(self, src, tgt):
        memory, pad = self.encode(src)
        return self.decode(tgt, memory, pad)

    @torch.no_grad()
    def greedy(self, src, bos_id, eos_id, max_new):
        self.eval()
        memory, pad = self.encode(src)          # encoder runs once
        out = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
        for _ in range(max_new):
            nxt = self.decode(out, memory, pad)[:, -1].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
            if (out == eos_id).any(dim=1).all():
                break
        return out
