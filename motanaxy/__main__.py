"""MOTANAXY CLI — Command-line interface for the training system.

Usage:
    python -m motanaxy ingest   --sources ./src ./docs --out corpus.txt
    python -m motanaxy train    --corpus corpus.txt --preset nano --steps 500
    python -m motanaxy generate --checkpoint runs/mtn_001/best_model.pt --prompt "def "
    python -m motanaxy info     --checkpoint runs/mtn_001/best_model.pt
    python -m motanaxy check
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from .ingestor import DataIngestor, build_corpus, corpus_stats
from .model import MotanaxyModel, PRESETS, from_preset, load_checkpoint
from .trainer import Trainer, TrainingConfig, prepare_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("motanaxy")


def cmd_ingest(args):
    """Ingest files/directories into a training corpus."""
    ingestor = DataIngestor(use_markitdown=not args.no_markitdown)
    documents = ingestor.ingest_paths(args.sources)

    if not documents:
        logger.error("No documents ingested. Check your --sources paths.")
        sys.exit(1)

    corpus = build_corpus(documents)
    stats = corpus_stats(corpus)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(corpus, encoding="utf-8")

    print(f"\n{'='*50}")
    print(f"  MOTANAXY Ingestion Complete")
    print(f"{'='*50}")
    print(f"  Files scanned:   {ingestor.stats['files_scanned']}")
    print(f"  Files ingested:  {ingestor.stats['files_ingested']}")
    print(f"  Files skipped:   {ingestor.stats['files_skipped']}")
    print(f"  Duplicates:      {ingestor.stats['files_duplicate']}")
    print(f"  Raw bytes:       {ingestor.stats['bytes_raw']:,}")
    print(f"  Clean bytes:     {ingestor.stats['bytes_clean']:,}")
    print(f"  Compression:     {(1 - ingestor.stats['bytes_clean'] / max(ingestor.stats['bytes_raw'], 1)) * 100:.1f}%")
    print(f"  Documents:       {stats['documents']}")
    print(f"  Corpus bytes:    {stats['total_bytes']:,}")
    print(f"  Unique bytes:    {stats['unique_bytes']}")
    print(f"  Saved to:        {out}")
    print(f"{'='*50}")


def cmd_train(args):
    """Train a MOTANAXY model."""
    # Load corpus
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        logger.error("Corpus not found: %s", corpus_path)
        logger.info("Run 'python -m motanaxy ingest' first to create a corpus.")
        sys.exit(1)

    corpus = corpus_path.read_text(encoding="utf-8")
    stats = corpus_stats(corpus)
    logger.info("Loaded corpus: %d documents, %d bytes", stats["documents"], stats["total_bytes"])

    # Prepare data splits
    train_data, val_data, test_data = prepare_data(
        corpus, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
    )
    logger.info("Splits — train: %d bytes, val: %d bytes, test: %d bytes",
                len(train_data), len(val_data), len(test_data))

    # Create model
    if args.preset:
        model = from_preset(args.preset)
        logger.info("Model preset: %s", args.preset)
    else:
        model = MotanaxyModel(
            width=args.width, context=args.context,
            layers=args.layers, heads=args.heads,
        )
        logger.info("Custom model: width=%d ctx=%d layers=%d heads=%d",
                     args.width, args.context, args.layers, args.heads)

    logger.info("Parameters: %s", f"{model.num_parameters:,}")

    # Training config
    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.steps,
        warmup_steps=args.warmup,
        grad_accum_steps=args.grad_accum,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        seed=args.seed,
    )

    # Output directory (auto-increment run number)
    out_dir = Path(args.out)
    if out_dir.exists() and not args.resume:
        # Find next available run number
        base = out_dir.parent
        prefix = out_dir.name
        i = 1
        while (base / f"{prefix}_{i:03d}").exists():
            i += 1
        out_dir = base / f"{prefix}_{i:03d}"
        logger.info("Output directory already exists, using: %s", out_dir)

    # Train
    trainer = Trainer(model, train_data, val_data, config, out_dir)

    # Resume
    if args.resume and (out_dir / "final_model.pt").exists():
        ckpt = torch.load(out_dir / "final_model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state"])
        if "optimizer" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer"])
        trainer.step = ckpt.get("step", 0)
        logger.info("Resumed from step %d", trainer.step)

    report = trainer.train()

    # Save corpus stats alongside report
    report["corpus_stats"] = stats
    report["model_preset"] = args.preset
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def cmd_generate(args):
    """Generate text from a trained model."""
    model = load_checkpoint(args.checkpoint)
    logger.info("Loaded model: %s parameters", f"{model.num_parameters:,}")

    prompt_bytes = list(args.prompt.encode("utf-8"))
    result_bytes = model.generate(
        prompt_bytes,
        max_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    output = bytes(result_bytes).decode("utf-8", errors="replace")
    print(output)


def cmd_info(args):
    """Show information about a checkpoint."""
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ckpt["config"]
    model = MotanaxyModel(**config)
    model.load_state_dict(ckpt["state"])

    print(f"\n{'='*50}")
    print(f"  MOTANAXY Model Info")
    print(f"{'='*50}")
    print(f"  Width:       {config['width']}")
    print(f"  Context:     {config['context']}")
    print(f"  Layers:      {config['layers']}")
    print(f"  Heads:       {config['heads']}")
    print(f"  Parameters:  {model.num_parameters:,}")
    print(f"  Size (est):  {model.num_parameters * 4 / 1024:.1f} KB (FP32)")
    print(f"{'='*50}")

    # Show report if available
    report_path = Path(args.checkpoint).parent / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"\n  Training Report:")
        for k, v in report.items():
            if k != "corpus_stats":
                print(f"    {k}: {v}")


def cmd_check(args):
    """Self-check: verify model correctness."""
    from .model import MotanaxyModel
    import torch

    print("Running MOTANAXY self-check...")

    # 1) Causal mask test
    model = MotanaxyModel(width=16, context=16, layers=1, heads=2).eval()
    x = torch.randint(256, (2, 12))
    other = x.clone()
    other[:, 6:] = torch.randint(256, (2, 6))
    assert torch.allclose(model(x)[:, :6], model(other)[:, :6], atol=1e-5), \
        "FAIL: Future token leakage detected"
    print("  [OK] Causal mask: no future leakage")

    # 2) Learning test
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    target = (x + 1) % 256
    first = torch.nn.functional.cross_entropy(
        model(x).reshape(-1, 256), target.reshape(-1)
    ).item()
    for _ in range(30):
        loss = torch.nn.functional.cross_entropy(
            model(x).reshape(-1, 256), target.reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first / 2, "FAIL: Model did not learn"
    print("  [OK] Gradient learning: loss decreased")

    # 3) Save/load test
    clone = MotanaxyModel(**model.config)
    clone.load_state_dict(model.state_dict())
    model.eval()
    clone.eval()
    assert torch.allclose(model(x), clone(x), atol=1e-5), \
        "FAIL: State reload mismatch"
    print("  [OK] State reload: weights preserved")

    # 4) Generation test
    ids = model.generate(list(b"def "), max_tokens=10, temperature=1.0)
    assert len(ids) == 14  # 4 prompt + 10 generated
    print("  [OK] Generation: produces valid output")

    print("\nPASS: All MOTANAXY checks passed!")


def main():
    parser = argparse.ArgumentParser(
        prog="motanaxy",
        description="MOTANAXY — Token-Efficient Code Training System",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- ingest ---
    p_ingest = sub.add_parser("ingest", help="Ingest files into training corpus")
    p_ingest.add_argument("--sources", nargs="+", required=True,
                          help="Files/directories to ingest")
    p_ingest.add_argument("--out", default="corpus.txt",
                          help="Output corpus file (default: corpus.txt)")
    p_ingest.add_argument("--no-markitdown", action="store_true",
                          help="Disable MarkItDown for non-text files")

    # --- train ---
    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--corpus", default="corpus.txt",
                         help="Path to training corpus")
    p_train.add_argument("--preset", choices=list(PRESETS), default=None,
                         help="Model size preset")
    p_train.add_argument("--width", type=int, default=64)
    p_train.add_argument("--context", type=int, default=128)
    p_train.add_argument("--layers", type=int, default=2)
    p_train.add_argument("--heads", type=int, default=4)
    p_train.add_argument("--steps", type=int, default=1000,
                         help="Training steps (default: 1000)")
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=3e-4,
                         help="Peak learning rate")
    p_train.add_argument("--warmup", type=int, default=50,
                         help="Warmup steps")
    p_train.add_argument("--grad-accum", type=int, default=1,
                         help="Gradient accumulation steps")
    p_train.add_argument("--eval-interval", type=int, default=100)
    p_train.add_argument("--save-interval", type=int, default=200)
    p_train.add_argument("--val-ratio", type=float, default=0.1)
    p_train.add_argument("--test-ratio", type=float, default=0.05)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--out", default="runs/mtn",
                         help="Output directory (auto-increments)")
    p_train.add_argument("--resume", action="store_true",
                         help="Resume training from existing checkpoint")

    # --- generate ---
    p_gen = sub.add_parser("generate", help="Generate text from model")
    p_gen.add_argument("--checkpoint", required=True,
                       help="Path to model checkpoint")
    p_gen.add_argument("--prompt", default="def ",
                       help="Generation prompt")
    p_gen.add_argument("--tokens", type=int, default=256,
                       help="Max tokens to generate")
    p_gen.add_argument("--temperature", type=float, default=0.7)
    p_gen.add_argument("--top-k", type=int, default=40,
                       help="Top-k sampling (0 = disabled)")

    # --- info ---
    p_info = sub.add_parser("info", help="Show checkpoint info")
    p_info.add_argument("--checkpoint", required=True)

    # --- check ---
    sub.add_parser("check", help="Run self-checks")

    args = parser.parse_args()
    try:
        {
            "ingest": cmd_ingest,
            "train": cmd_train,
            "generate": cmd_generate,
            "info": cmd_info,
            "check": cmd_check,
        }[args.command](args)
    except (ValueError, OSError) as exc:
        logger.error("Error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
