# Transformers from scratch: GPT, BERT, encoder-decoder, BPE, and every common kind of attention

Hand-written PyTorch implementations of the Transformer family, with no `nn.Transformer` and no `scaled_dot_product_attention`. The repo covers all three Transformer families (decoder-only, encoder-only and encoder-decoder) and a byte-level BPE tokenizer. All three models are built from **one** attention module in which every variant is a flag.

| File | What's in it |
|---|---|
| [`transformer.py`](transformer.py) | `Attention`, `Block`, `MiniGPT` (decoder-only), `BERT` (encoder-only), `Seq2Seq` (encoder-decoder) |
| [`bpe.py`](bpe.py) | Byte-level BPE tokenizer (the GPT-2 algorithm): train, encode, decode, with no unknown tokens |
| [`slm_minigpt.ipynb`](slm_minigpt.ipynb) | GPT trained on Dostoevsky, Kafka, Plato, Shakespeare, Blake, with a char or BPE tokenizer. Covers sampling, KV cache, attention heatmaps, MHA/GQA/MQA |
| [`bert.ipynb`](bert.ipynb) | BERT with masked language modelling on the same corpus, using BPE tokens. Fill-in-the-blank, bidirectional context, padding mask |
| [`seq2seq.ipynb`](seq2seq.ipynb) | Encoder-decoder that normalizes dates (`"Saturday, 31 October 2026"` → `2026-10-31`). **99.7% exact match on unseen dates**, with cross-attention alignment plots |
| [`tests.py`](tests.py) | Behavioural tests: each mask blocks what it should, the KV cache matches a full forward pass, BERT sees both directions, BPE round-trips any string |

## Concepts → code

| Concept | Where | How it's checked |
|---|---|---|
| Scaled dot-product attention $\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ | `Attention.forward` | |
| Multi-head attention (split → attend → concatenate → project) | `Attention.forward` | |
| **Decoder** self-attention (causal mask) | `Attention(causal=True)` | `tests.py::test_causal_cannot_see_future` |
| **Encoder** self-attention (bidirectional) | `Attention(causal=False)` | `tests.py::test_encoder_sees_both_directions` |
| **Cross-attention** (Q from decoder, K/V from encoder) | `Attention.forward(x, memory=...)` | `tests.py::test_cross_attention_reads_memory`, heatmaps in `seq2seq.ipynb` |
| Padding mask | `key_padding_mask` | `tests.py::test_padding_is_ignored` |
| Sliding-window (local) attention | `Attention(window=w)` | `tests.py::test_sliding_window` |
| Grouped-query / multi-query attention | `Attention(n_kv_heads=...)` | `tests.py::test_gqa_mqa_shrink_kv` |
| KV cache | `Attention.forward(cache=...)`, `MiniGPT.generate` | `tests.py::test_kv_cache_matches_full_forward`, timing in `slm_minigpt.ipynb` |
| Pre-LN blocks, residuals, GELU MLP | `Block` | |
| Decoder-only LM (GPT) | `MiniGPT` | |
| Encoder-only (BERT), masked LM with the 80/10/10 rule | `BERT`, `bert.ipynb` | `tests.py::test_bert_is_bidirectional_and_ignores_padding` |
| Encoder-decoder (*Attention Is All You Need*) | `Seq2Seq` | `seq2seq.ipynb` |
| Byte-level BPE tokenizer | `bpe.py` | `tests.py::test_bpe_round_trip_and_merges` |
| Temperature + top-k sampling, greedy decoding | `MiniGPT.generate`, `Seq2Seq.greedy` | |

## Run it

```bash
pip install -r requirements.txt
python tests.py                   # 11 checks, a few seconds
jupyter notebook                  # seq2seq ~4 min and bert ~15 min on CPU; slm_minigpt wants a GPU
```

## Next steps

- RoPE positional encoding, which would also allow a rolling KV cache past `context_length`
