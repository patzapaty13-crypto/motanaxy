"""MOTANAXY Minimal BPE Tokenizer.

A pure-Python Byte-Pair Encoding (BPE) tokenizer.
Trains a vocabulary to compress frequent byte sequences (like 'def ', 'return ').
"""

import json
import random
from pathlib import Path


def get_stats(ids):
    """Count frequencies of adjacent pairs."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """Replace all consecutive occurrences of pair with the new idx."""
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


class BPETokenizer:
    def __init__(self):
        self.merges = {}  # (int, int) -> int
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int, verbose=False):
        """Train the tokenizer on a given text."""
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        
        # Convert text to raw bytes, then to list of ints
        ids = list(text.encode("utf-8"))
        
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            
            # Find the most frequent pair
            best_pair = max(stats, key=stats.get)
            idx = 256 + i
            
            # Record merge and update vocab
            self.merges[best_pair] = idx
            self.vocab[idx] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # Apply merge to the data
            ids = merge(ids, best_pair, idx)
            
            if verbose and (i + 1) % 500 == 0:
                print(f"Merge {i+1}/{num_merges}: {best_pair} -> {idx} ({self.vocab[idx]!r})")

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token ids."""
        ids = list(text.encode("utf-8"))
        
        # We need to apply merges in the exact order they were learned
        # (This is a naive O(N^2) encode, but fine for short prompts)
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Find the pair that was merged earliest
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # No more merges possible
                
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
            
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into a string."""
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")

    def save(self, path: str):
        """Save the tokenizer merges to a file."""
        # Convert tuple keys to strings for JSON
        merges_str = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        with open(path, "w") as f:
            json.dump(merges_str, f)

    def load(self, path: str):
        """Load tokenizer merges from a file."""
        with open(path, "r") as f:
            merges_str = json.load(f)
            
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        for k_str, idx in merges_str.items():
            p1, p2 = map(int, k_str.split(","))
            self.merges[(p1, p2)] = idx
            self.vocab[idx] = self.vocab[p1] + self.vocab[p2]


if __name__ == "__main__":
    # Test script: train a small tokenizer on our synthetic data
    import json
    import time
    
    print("Loading synthetic data for tokenizer training...")
    with open("training_data/synthetic_10k.jsonl") as f:
        lines = f.readlines()
        
    # Sample ~250KB of data to train on (keeps training fast <10 seconds)
    # Using the whole 3.4MB with naive O(N^2) would take too long
    random.seed(42)
    sample = "".join(json.loads(line)["text"] for line in random.sample(lines, 500))
    
    print(f"Training tokenizer on {len(sample.encode('utf-8'))/1024:.1f} KB sample...")
    start = time.perf_counter()
    
    tokenizer = BPETokenizer()
    tokenizer.train(sample, vocab_size=4096, verbose=True)
    
    tokenizer.save("motanaxy/tokenizer.json")
    print(f"Tokenizer trained in {time.perf_counter() - start:.1f}s and saved to tokenizer.json")
    
    # Test encoding
    test_str = "def sum_list(lst):\n    return sum(lst)\n"
    ids = tokenizer.encode(test_str)
    print(f"\nOriginal ({len(test_str.encode('utf-8'))} bytes): {repr(test_str)}")
    print(f"Encoded ({len(ids)} tokens): {ids}")
    print(f"Decoded: {repr(tokenizer.decode(ids))}")
    print(f"Compression ratio: {len(test_str.encode('utf-8')) / len(ids):.2f}x")
