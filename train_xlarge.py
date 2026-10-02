"""Train MOTANAXY xlarge (10M params) with BPE Tokenizer on 10k Synthetic dataset.

This script implements the "Reddit advice" architecture:
1. ~10M parameter model (xlarge preset: 12 layers, 8 heads, 256 width)
2. BPE Tokenizer (vocab size 4096) for better context utilization
3. 10,000+ verified synthetic algorithmic examples
"""

import argparse
import ast
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from motanaxy.model import from_preset, load_checkpoint
from motanaxy.tokenizer import BPETokenizer
from motanaxy.trainer import DataLoader, Trainer, TrainingConfig


# ---------------------------------------------------------------------------
# Data preparation using Tokenizer
# ---------------------------------------------------------------------------

def load_and_tokenize_data(tokenizer: BPETokenizer, data_path: str):
    """Load jsonl records, split into train/val/test, and tokenize them."""
    print(f"Loading data from {data_path}...")
    
    with open(data_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
        
    # Shuffle predictably
    import random
    random.Random(42).shuffle(records)
    
    # 9000 train, 500 val, 500 test
    train_recs = records[:-1000]
    val_recs = records[-1000:-500]
    test_recs = records[-500:]
    
    print("Tokenizing datasets (this is fast because we process document-by-document)...")
    
    def tokenize_records(recs):
        tokens = []
        for r in recs:
            # Encode each document and add separator
            encoded = tokenizer.encode(r["text"] + "\n\n\n")
            tokens.extend(encoded)
        return torch.tensor(tokens, dtype=torch.long)
        
    train_tensor = tokenize_records(train_recs)
    val_tensor = tokenize_records(val_recs)
    
    return train_tensor, val_tensor, test_recs


# ---------------------------------------------------------------------------
# Assessment (Tokenizer Aware)
# ---------------------------------------------------------------------------

@torch.no_grad()
def assess(model, tokenizer, test_records, num_samples=10):
    """Evaluate on held-out test records using the tokenizer."""
    model.eval()
    samples = []
    
    # Just sample a few to save time during evaluation
    eval_records = test_records[:num_samples]
    
    for record in eval_records:
        prompt = record["prompt"]
        prompt_ids = tokenizer.encode(prompt)
        
        torch.manual_seed(123)
        # Generate max 100 tokens (since 1 token ~ 3-4 chars, 100 is plenty)
        output_ids = model.generate(
            prompt_ids, 
            max_tokens=100,
            temperature=0.8, 
            top_k=40
        )
        
        output_text = tokenizer.decode(output_ids)
        continuation = output_text[len(prompt):]
        candidate = continuation.split("\n\n", 1)[0].strip()
        
        valid = False
        try:
            tree = ast.parse(candidate)
            valid = bool(tree.body) and isinstance(tree.body[0], ast.FunctionDef)
        except (SyntaxError, ValueError):
            pass
            
        samples.append(dict(
            concept=record["concept"],
            prompt=prompt, 
            output=continuation,
            candidate=candidate, 
            parses_as_function=valid,
        ))
        
    return dict(
        syntactically_valid_functions=sum(s["parses_as_function"] for s in samples),
        prompts=len(samples), 
        samples=samples,
    )


# ---------------------------------------------------------------------------
# Main training entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/xlarge_10M",
                        help="Output directory")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Number of training steps")
    parser.add_argument("--data", default="training_data/synthetic_10k.jsonl",
                        help="Path to synthetic jsonl dataset")
    parser.add_argument("--tokenizer", default="motanaxy/tokenizer.json",
                        help="Path to trained BPE tokenizer")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        parser.error(f"Output exists: {out}. Choose a new --out.")

    torch.set_num_threads(8)
    torch.manual_seed(42)

    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)
    vocab_size = len(tokenizer.vocab)
    print(f"Tokenizer loaded. Vocab size: {vocab_size}")

    # 2. Load and Tokenize Data
    train_data, val_data, test_records = load_and_tokenize_data(tokenizer, args.data)
    print(f"Training tokens: {len(train_data):,} (compressed from bytes)")
    print(f"Validation tokens: {len(val_data):,} (compressed from bytes)")

    # 3. Initialize Model (10M params)
    print("\nInitializing xlarge model (10M parameters)...")
    # xlarge has vocab_size=4096 built-in, but we enforce it just in case
    model = from_preset("xlarge")
    model.config["vocab_size"] = vocab_size
    print(f"Config: {model.config}")
    print(f"Total Parameters: {model.num_parameters:,}")

    out.mkdir(parents=True)

    # 4. Baseline Assessment
    print("\nRunning baseline assessment...")
    before = assess(model, tokenizer, test_records)
    print(f"Baseline syntax: {before['syntactically_valid_functions']}/{before['prompts']}")

    # 5. Configure Training
    # Since 10M params is heavy on CPU, we use smaller batch size but accumulate gradients
    config = TrainingConfig(
        max_steps=args.steps,
        learning_rate=3e-4,
        warmup_steps=500,
        batch_size=4,             # Small batch to fit CPU cache efficiently
        grad_accum_steps=4,       # Effective batch size = 16
        eval_interval=500,
        eval_batches=16,
        save_interval=500,
        device="cpu",
        use_amp=False,
        weight_decay=0.1,
    )

    # 6. Train
    print(f"\nStarting training: {args.steps} steps...")
    started = time.perf_counter()
    trainer = Trainer(model, train_data, val_data, config, out)
    trainer.train()
    elapsed = time.perf_counter() - started
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/3600:.2f} hours)")

    # 7. Final Assessment
    print("\nFinal assessment...")
    best = load_checkpoint(str(out / "best_model.pt"))
    after = assess(best, tokenizer, test_records, num_samples=20)
    
    print(f"Final syntax: {after['syntactically_valid_functions']}/{after['prompts']}")
    
    # Save results
    comparison = dict(
        before=before,
        after=after,
        best_step=torch.load(out / "best_model.pt", weights_only=True)["step"],
        architecture="xlarge (10M) + BPE Tokenizer",
        parameters=model.num_parameters,
    )
    (out / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
        
    print(f"Results saved to {out.resolve()}")


if __name__ == "__main__":
    main()
