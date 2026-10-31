# Transformers from scratch: decoder, encoder, and every common kind of attention

Hand-written PyTorch implementations of the Transformer family, with no `nn.Transformer` and no `scaled_dot_product_attention`. The repo contains a GPT-style language model and a full encoder-decoder, built from **one** attention module in which every variant is a flag.

| File | What's in it |
|---|---|
| [`transformer.py`](transformer.py) | `Attention`, `Block`, `MiniGPT` (decoder-only), `Seq2Seq` (encoder-decoder) |
| [`slm_minigpt.ipynb`](slm_minigpt.ipynb) | Character-level GPT trained on Dostoevsky, Kafka, Plato, Shakespeare, Blake. Covers sampling, KV cache, attention heatmaps, MHA/GQA/MQA |
| [`seq2seq.ipynb`](seq2seq.ipynb) | Encoder-decoder that normalizes dates (`"Saturday, 31 October 2026"` → `2026-10-31`). **99.7% exact match on unseen dates**, with cross-attention alignment plots |
| [`test_transformer.py`](test_transformer.py) | Behavioural tests: each mask blocks what it should, and the KV cache matches a full forward pass |

## Concepts → code

| Concept | Where | How it's checked |
|---|---|---|
| Scaled dot-product attention $\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ | `Attention.forward` | |
| Multi-head attention (split → attend → concatenate → project) | `Attention.forward` | |
| **Decoder** self-attention (causal mask) | `Attention(causal=True)` | `test_causal_cannot_see_future` |
| **Encoder** self-attention (bidirectional) | `Attention(causal=False)` | `test_encoder_sees_both_directions` |
| **Cross-attention** (Q from decoder, K/V from encoder) | `Attention.forward(x, memory=...)` | `test_cross_attention_reads_memory`, heatmaps in `seq2seq.ipynb` |
| Padding mask | `key_padding_mask` | `test_padding_is_ignored` |
| Sliding-window (local) attention | `Attention(window=w)` | `test_sliding_window` |
| Grouped-query / multi-query attention | `Attention(n_kv_heads=...)` | `test_gqa_mqa_shrink_kv` |
| KV cache | `Attention.forward(cache=...)`, `MiniGPT.generate` | `test_kv_cache_matches_full_forward`, timing in `slm_minigpt.ipynb` |
| Pre-LN blocks, residuals, GELU MLP | `Block` | |
| Decoder-only LM (GPT) | `MiniGPT` | |
| Encoder-decoder (*Attention Is All You Need*) | `Seq2Seq` | `seq2seq.ipynb` |
| Temperature + top-k sampling, greedy decoding | `MiniGPT.generate`, `Seq2Seq.greedy` | |

## Run it

```bash
pip install -r requirements.txt
python test_transformer.py        # 9 checks, a few seconds
jupyter notebook                  # seq2seq.ipynb trains in ~4 min on CPU; slm_minigpt.ipynb wants a GPU
```

## Next steps

- RoPE positional encoding, which would also allow a rolling KV cache past `context_length`
- BPE tokenization instead of characters
- An encoder-only (BERT-style) masked-language-model objective
