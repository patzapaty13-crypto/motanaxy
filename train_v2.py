"""Train MOTANAXY v2 (large preset) from random weights on all available data.

Usage:
    .\\.venv\\Scripts\\python.exe train_v2.py --check
    .\\.venv\\Scripts\\python.exe train_v2.py --steps 5000 --out runs/v2_large
"""
import argparse
import ast
import hashlib
import inspect
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from motanaxy.model import MotanaxyModel, from_preset, load_checkpoint
from motanaxy.trainer import DataLoader, Trainer, TrainingConfig
from training_data import verified_lessons as lessons


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_corpus_data(corpus_path: str) -> str:
    """Load raw corpus text, splitting by triple-newline."""
    return Path(corpus_path).read_text(encoding="utf-8")


def prepare_lesson_records():
    """Prepare verified_lessons as structured records with train/val/test split."""
    import random
    checks = lessons.check()
    names = sorted(lessons.CASES)
    random.Random(2026).shuffle(names)
    groups = {"test": names[:6], "validation": names[6:12], "train": names[12:]}
    records = {}
    for split, members in groups.items():
        records[split] = []
        for name in members:
            function = getattr(lessons, name)
            source = inspect.getsource(function)
            tree = ast.parse(source)
            node = tree.body[0]
            english, thai = inspect.getdoc(function).split(" | ")
            del node.body[0]
            code = ast.unparse(tree) + "\n"
            examples = []
            for arguments, expected in lessons.CASES[name]:
                arguments_text = ", ".join(repr(arg) for arg in arguments)
                examples.append(f"assert {name}({arguments_text}) == {expected!r}")
            for language, description in [("en", english), ("th", thai)]:
                document = f"# {description}\n{code}\n" + "\n".join(examples)
                ast.parse(document)
                records[split].append(dict(
                    concept=name, language=language,
                    prompt=f"# {description}\n", code=code, text=document,
                ))
    return groups, records, checks


def build_training_tensor(records: list[dict], corpus_path: str = None) -> torch.Tensor:
    """Combine lesson records + optional corpus into a single training tensor."""
    parts = []
    # Add structured lessons
    lesson_text = "\n\n\n".join(r["text"] for r in records)
    parts.append(lesson_text)

    # Add corpus data if available
    if corpus_path and Path(corpus_path).exists():
        corpus = Path(corpus_path).read_text(encoding="utf-8")
        parts.append(corpus)

    combined = "\n\n\n".join(parts)
    return torch.tensor(list(combined.encode("utf-8")), dtype=torch.long)


def tensor_from_records(records: list[dict]) -> torch.Tensor:
    """Convert records to a byte tensor (for val/test)."""
    text = "\n\n\n".join(r["text"] for r in records)
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)


# ---------------------------------------------------------------------------
# Assessment (same protocol as train_knowledge.py for comparison)
# ---------------------------------------------------------------------------

@torch.no_grad()
def assess(model, test_records):
    """Evaluate on held-out test records. No generated code is executed."""
    model.eval()
    data = tensor_from_records(test_records)
    ctx = model.config["context"]
    loader = DataLoader(data, ctx, 4, seed=912)
    losses = []
    for _ in range(8):
        x, y = loader.sample()
        losses.append(
            F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1)).item()
        )
    samples = []
    for record in test_records:
        prompt = record["prompt"]
        torch.manual_seed(123)
        output = bytes(model.generate(
            list(prompt.encode()), max_tokens=180,
            temperature=1.0, top_k=1,
        )).decode("utf-8", errors="replace")
        continuation = output[len(prompt):]
        candidate = continuation.split("\n\n", 1)[0].strip()
        valid = False
        try:
            tree = ast.parse(candidate)
            valid = bool(tree.body) and isinstance(tree.body[0], ast.FunctionDef)
        except (SyntaxError, ValueError):
            pass
        samples.append(dict(
            concept=record["concept"], language=record["language"],
            prompt=prompt, output=continuation,
            candidate=candidate, parses_as_function=valid,
        ))
    return dict(
        loss=sum(losses) / len(losses),
        syntactically_valid_functions=sum(s["parses_as_function"] for s in samples),
        prompts=len(samples), samples=samples,
        limitation="Syntax only; no generated-code execution or correctness claim.",
    )


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def self_check():
    """Verify v2 architecture: causal mask, learning, checkpoint reload, KV-cache."""
    print("Testing v2 architecture...", flush=True)

    # 1. Create v2 model
    m = MotanaxyModel(width=48, context=64, layers=2, heads=6, version=2, dropout=0.0)
    m.eval()
    print(f"  Test model: {m.num_parameters:,} parameters", flush=True)

    # 2. Causal mask test
    x = torch.randint(256, (2, 32))
    x2 = x.clone()
    x2[:, 16:] = torch.randint(256, (2, 16))
    out1 = m(x)[:, :16]
    out2 = m(x2)[:, :16]
    assert torch.allclose(out1, out2, atol=1e-5), "FAIL: Future token leakage"
    print("  PASS: Causal mask", flush=True)

    # 3. Learning test
    optimizer = torch.optim.AdamW(m.parameters(), lr=0.01)
    target = (x + 1) % 256
    first = F.cross_entropy(m(x).reshape(-1, 256), target.reshape(-1)).item()
    for _ in range(50):
        m.train()
        loss = F.cross_entropy(m(x).reshape(-1, 256), target.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < first * 0.7, f"FAIL: Loss did not decrease enough ({first:.4f} -> {loss.item():.4f})"
    print(f"  PASS: Learning ({first:.4f} -> {loss.item():.4f})", flush=True)

    # 4. Checkpoint save/reload test
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.pt"
        torch.save({"config": m.config, "state": m.state_dict()}, path)
        m2 = load_checkpoint(str(path))
        m.eval()
        m2.eval()
        assert torch.allclose(m(x), m2(x), atol=1e-5), "FAIL: Checkpoint reload mismatch"
    print("  PASS: Checkpoint reload", flush=True)

    # 5. KV-cache generation test
    m.eval()
    prompt = list(b"def f(x):")
    result = m.generate(prompt, max_tokens=20, temperature=0.7, top_k=10)
    assert len(result) == len(prompt) + 20, f"FAIL: Expected {len(prompt)+20} bytes, got {len(result)}"
    print("  PASS: KV-cache generation", flush=True)

    # 6. KV-cache consistency: cached vs non-cached generation must match (greedy)
    torch.manual_seed(42)
    result_cached = m.generate(prompt, max_tokens=30, temperature=1.0, top_k=1)
    # Compare with v1-style (no cache) on same model — re-implement without cache
    torch.manual_seed(42)
    ids = list(prompt)
    ctx = m.config["context"]
    for _ in range(30):
        window = ids[-ctx:]
        xt = torch.tensor([window])
        logits = m(xt)[0, -1] / 1.0
        top_vals, _ = logits.topk(1)
        logits[logits < top_vals[-1]] = float("-inf")
        probs = logits.softmax(-1)
        ids.append(torch.multinomial(probs, 1).item())
    assert result_cached == ids, "FAIL: KV-cache output differs from non-cached"
    print("  PASS: KV-cache consistency (cached == non-cached)", flush=True)

    # 7. Verify backward compat: v1 model still works
    m1 = MotanaxyModel(width=32, context=32, layers=1, heads=4, version=1)
    m1.eval()
    x1 = torch.randint(256, (1, 16))
    _ = m1(x1)
    _ = m1.generate(list(b"def "), max_tokens=10)
    print("  PASS: v1 backward compatibility", flush=True)

    # 8. DataLoader edge case
    data = torch.arange(9)
    loader = DataLoader(data, 8, 1)
    xb, yb = loader.sample()
    assert xb.tolist() == [list(range(8))] and yb.tolist() == [list(range(1, 9))]
    print("  PASS: DataLoader sampling", flush=True)

    print("\nAll v2 checks passed!", flush=True)


# ---------------------------------------------------------------------------
# Main training entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="large",
                        help="Model preset: nano, micro, small, medium, large (default: large)")
    parser.add_argument("--out", default="runs/v2_large",
                        help="Output directory for checkpoints and metrics")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Number of training steps")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate")
    parser.add_argument("--warmup", type=int, default=200,
                        help="Warmup steps")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--corpus", default="corpus_full.txt",
                        help="Path to additional corpus data")
    parser.add_argument("--check", action="store_true",
                        help="Run self-check and exit")
    args = parser.parse_args()

    if args.check:
        self_check()
        return

    if args.steps < 1:
        parser.error("--steps must be positive")

    out = Path(args.out)
    if out.exists():
        parser.error(f"Output exists: {out}. Choose a new --out to preserve earlier experiments.")

    torch.set_num_threads(4)  # Use more cores for larger model
    torch.manual_seed(42)

    # Prepare data
    print("Preparing data...", flush=True)
    groups, records, checks = prepare_lesson_records()
    print(f"Verified lessons: {len(lessons.CASES)} concepts, {checks} reference checks", flush=True)

    # Build training data: lessons + corpus
    train_data = build_training_tensor(records["train"], args.corpus)
    val_data = tensor_from_records(records["validation"])
    print(f"Training data: {len(train_data):,} bytes", flush=True)
    print(f"Validation data: {len(val_data):,} bytes", flush=True)

    # Create model
    model = from_preset(args.preset)
    print(f"Model preset: {args.preset}", flush=True)
    print(f"Config: {model.config}", flush=True)
    print(f"Parameters: {model.num_parameters:,}", flush=True)
    print(f"Architecture: v{model.version} (RMSNorm + RoPE + SwiGLU)", flush=True)

    # Save manifest
    out.mkdir(parents=True)
    manifest = dict(
        preset=args.preset,
        config=model.config,
        parameters=model.num_parameters,
        architecture_version=model.version,
        source="verified_lessons.py + corpus_full.txt",
        reference_checks=checks,
        groups=groups,
        documents={split: len(rows) for split, rows in records.items()},
        source_sha256=hashlib.sha256(Path(lessons.__file__).read_bytes()).hexdigest(),
        corpus_sha256=hashlib.sha256(
            Path(args.corpus).read_bytes()).hexdigest() if Path(args.corpus).exists() else None,
        training_seed=42,
        split_seed=2026,
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save data splits
    for split, entries in records.items():
        (out / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in entries) + "\n",
            encoding="utf-8",
        )

    # Baseline assessment
    print("\nBaseline assessment (random weights)...", flush=True)
    before = assess(model, records["test"])
    (out / "baseline.json").write_text(
        json.dumps(before, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Baseline: loss={before['loss']:.4f}, "
          f"syntax={before['syntactically_valid_functions']}/{before['prompts']}", flush=True)

    # Training config
    config = TrainingConfig(
        max_steps=args.steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        batch_size=args.batch_size,
        eval_interval=250,
        eval_batches=8,
        save_interval=1000,
        device="cpu",
        use_amp=False,
        weight_decay=0.1,
        grad_clip=1.0,
    )

    # Train
    print(f"\nStarting training: {args.steps} steps...", flush=True)
    started = time.perf_counter()
    trainer = Trainer(model, train_data, val_data, config, out)
    assert trainer.evaluate() == trainer.evaluate(), "validation windows must stay fixed"
    trainer.train()
    elapsed = time.perf_counter() - started
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    # Final assessment
    print("\nFinal assessment...", flush=True)
    best = load_checkpoint(str(out / "best_model.pt"))
    after = assess(best, records["test"])

    comparison = dict(
        before=before,
        after=after,
        best_step=torch.load(out / "best_model.pt", weights_only=True)["step"],
        heldout_loss_improved=after["loss"] < before["loss"],
        architecture="v2 (RMSNorm + RoPE + SwiGLU + Dropout)",
        parameters=model.num_parameters,
        preset=args.preset,
        note="New model from random weights. Does not inherit knowledge_v1 training.",
    )
    (out / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Final: loss={after['loss']:.4f}, "
          f"syntax={after['syntactically_valid_functions']}/{after['prompts']}", flush=True)
    print(f"Improvement: {before['loss'] - after['loss']:.4f} loss reduction", flush=True)
    print(f"Results: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
