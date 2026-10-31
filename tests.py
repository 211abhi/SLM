"""Behavioural checks for every attention variant, BERT and the BPE tokenizer. Run: python tests.py"""

import torch

from bpe import BPE, PRETOKENIZE
from transformer import BERT, Attention, MiniGPT, Seq2Seq

torch.manual_seed(0)
D, H = 32, 4


def close(a, b):
    return torch.allclose(a, b, atol=1e-5)


def test_causal_cannot_see_future():
    attn = Attention(D, H, causal=True).eval()
    x = torch.randn(1, 10, D)
    y = x.clone()
    y[:, 7] += 1.0                                  # change token 7
    assert close(attn(x)[:, :7], attn(y)[:, :7])    # tokens 0..6 unaffected
    assert not close(attn(x)[:, 7:], attn(y)[:, 7:])


def test_encoder_sees_both_directions():
    attn = Attention(D, H, causal=False).eval()
    x = torch.randn(1, 10, D)
    y = x.clone()
    y[:, 7] += 1.0
    assert not close(attn(x)[:, :7], attn(y)[:, :7])  # earlier tokens DO see token 7


def test_padding_is_ignored():
    attn = Attention(D, H).eval()
    x = torch.randn(1, 6, D)
    padded = torch.cat([x, torch.randn(1, 4, D)], dim=1)   # 4 garbage pad positions
    mask = torch.tensor([[False] * 6 + [True] * 4])
    assert close(attn(x), attn(padded, key_padding_mask=mask)[:, :6])


def test_sliding_window():
    attn = Attention(D, H, causal=True, window=3).eval()
    x = torch.randn(1, 10, D)
    y = x.clone()
    y[:, 2] += 1.0
    out_x, out_y = attn(x), attn(y)
    assert not close(out_x[:, 4], out_y[:, 4])   # 4 - 2 < 3: inside the window
    assert close(out_x[:, 5:], out_y[:, 5:])     # 5 - 2 >= 3: outside the window


def test_cross_attention_reads_memory():
    attn = Attention(D, H).eval()
    x = torch.randn(1, 5, D)
    m1, m2 = torch.randn(1, 8, D), torch.randn(1, 8, D)
    out = attn(x, memory=m1)
    assert out.shape == (1, 5, D)                # query length kept, memory length can differ
    assert not close(out, attn(x, memory=m2))


def test_kv_cache_matches_full_forward():
    for n_kv, window in [(4, None), (2, None), (1, None), (4, 3)]:   # MHA, GQA, MQA, sliding
        model = MiniGPT(20, 16, D, 2, H, n_kv_heads=n_kv, window=window).eval()
        idx = torch.randint(0, 20, (2, 12))
        full = model(idx)
        caches = [{} for _ in model.blocks]
        steps = [model(idx[:, :5], caches)]                           # prefill 5 tokens
        steps += [model(idx[:, t:t + 1], caches, start_pos=t) for t in range(5, 12)]
        assert close(full, torch.cat(steps, dim=1)), (n_kv, window)


def test_gqa_mqa_shrink_kv():
    params = [Attention(D, H, n_kv_heads=n).kv.weight.numel() for n in (4, 2, 1)]
    assert params[0] == 2 * params[1] == 4 * params[2]


def test_generate_runs_past_context():
    model = MiniGPT(20, 16, D, 2, H)
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), 40, top_k=5)
    assert out.shape == (1, 41)


def test_seq2seq_shapes_and_greedy():
    model = Seq2Seq(20, 32, D, 2, H, pad_id=0)
    src, tgt = torch.randint(1, 20, (3, 9)), torch.randint(1, 20, (3, 6))
    assert model(src, tgt).shape == (3, 6, 20)
    assert model.greedy(src, bos_id=1, eos_id=2, max_new=10).size(1) <= 11


def test_bert_is_bidirectional_and_ignores_padding():
    model = BERT(20, 16, D, 2, H, pad_id=0).eval()
    idx = torch.randint(1, 20, (1, 8))
    changed = idx.clone()
    changed[0, 6] = (idx[0, 6] % 19) + 1                           # change a LATER token
    assert not close(model(idx)[:, :6], model(changed)[:, :6])     # earlier outputs see it
    padded = torch.cat([idx, torch.zeros(1, 5, dtype=torch.long)], dim=1)
    assert close(model(idx), model(padded)[:, :8])                 # trailing pads change nothing


def test_bpe_round_trip_and_merges():
    corpus = "the cat sat on the mat. the cat ate the rat! " * 20
    tok = BPE.train(corpus, 300)
    assert 256 < tok.vocab_size <= 300                             # stops early once every word is one token
    for s in [corpus, "unseen wörds, émojis 🙂 and tabs\t\n", "_under_scores_", ""]:
        assert "".join(PRETOKENIZE.findall(s)) == s               # pre-tokenizer drops nothing
        assert tok.decode(tok.encode(s)) == s                     # byte-level: anything round-trips
    assert len(tok.encode(corpus)) < len(corpus.encode()) / 3     # frequent words collapse to few tokens
    assert tok.pieces(" the cat") == [" the", " cat"]


if __name__ == "__main__":
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} passed")
