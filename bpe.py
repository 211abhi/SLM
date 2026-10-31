"""
Byte-level Byte Pair Encoding, the tokenizer algorithm behind GPT-2/3/4, from scratch.

Training: start with the 256 possible bytes as tokens, then repeatedly find the most frequent
adjacent pair of tokens in the corpus and merge it into a new token, until vocab_size is reached.
Encoding: replay those merges, earliest-learned first. Because the base alphabet is raw bytes,
any string can be encoded (no unknown tokens), unlike the character tokenizer.
"""

import re
from collections import Counter

# split text into word-ish chunks first so merges never cross word boundaries ("dog." != "dog" + ".")
# (GPT-2 uses the same idea with \p{L}/\p{N} classes from the `regex` package)
PRETOKENIZE = re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+|.")


def merge(seq, pair, new_id):
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


class BPE:
    def __init__(self, merges):
        self.merges = {tuple(pair): 256 + rank for rank, pair in enumerate(merges)}  # pair -> new id
        self.vocab = {i: bytes([i]) for i in range(256)}                             # id -> bytes
        for (a, b), i in self.merges.items():
            self.vocab[i] = self.vocab[a] + self.vocab[b]
        self.cache = {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    @classmethod
    def train(cls, text, vocab_size):
        # work on unique words weighted by count: ~60k words instead of millions of characters
        words = [(list(w.encode("utf-8")), n) for w, n in Counter(PRETOKENIZE.findall(text)).items()]
        merges = []
        for new_id in range(256, vocab_size):
            pairs = Counter()
            for seq, n in words:
                for pair in zip(seq, seq[1:]):
                    pairs[pair] += n
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            merges.append(best)
            # ponytail: full recount every merge, O(merges x corpus); incremental pair counts if vocab > ~5k
            words = [(merge(seq, best, new_id), n) for seq, n in words if len(seq) > 1]
        return cls(merges)

    def encode(self, text):
        ids = []
        for word in PRETOKENIZE.findall(text):
            if word not in self.cache:
                self.cache[word] = self._encode_word(word)
            ids.extend(self.cache[word])
        return ids

    def _encode_word(self, word):
        seq = list(word.encode("utf-8"))
        while len(seq) > 1:
            # apply the merge that was learned earliest, exactly as in training
            pair = min(zip(seq, seq[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            seq = merge(seq, pair, self.merges[pair])
        return seq

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def pieces(self, text):
        """The text split into its tokens, for display."""
        return [self.vocab[i].decode("utf-8", errors="replace") for i in self.encode(text)]
