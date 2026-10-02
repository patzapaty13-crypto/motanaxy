"""MOTANAXY Trainer — Production-ready training loop.

Features:
- Cosine LR schedule with linear warmup
- Gradient accumulation (train larger effective batches on limited memory)
- Mixed-precision training (AMP) when GPU available
- Periodic validation & checkpointing
- Resume from checkpoint
- Detailed metrics logging (JSON)
"""

import hashlib
import json
import logging
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .model import MotanaxyModel

logger = logging.getLogger("motanaxy.trainer")


class TrainingConfig:
    """All training hyperparameters in one place."""

    def __init__(
        self,
        # Data
        batch_size: int = 8,
        # Optimization
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 50,
        max_steps: int = 1000,
        grad_clip: float = 1.0,
        grad_accum_steps: int = 1,
        # Schedule
        lr_schedule: str = "cosine",  # "cosine" or "constant"
        min_lr_ratio: float = 0.1,
        # Validation & checkpointing
        eval_interval: int = 100,
        eval_batches: int = 8,
        save_interval: int = 200,
        # System
        device: str = "auto",
        use_amp: bool = True,
        seed: int = 42,
        num_threads: int = 2,
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        self.grad_accum_steps = grad_accum_steps
        self.lr_schedule = lr_schedule
        self.min_lr_ratio = min_lr_ratio
        self.eval_interval = eval_interval
        self.eval_batches = eval_batches
        self.save_interval = save_interval
        self.device = device
        self.use_amp = use_amp
        self.seed = seed
        self.num_threads = num_threads

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class DataLoader:
    """Minimal, memory-efficient data loader for byte sequences.

    No external dependencies — just torch (Ponytail: standard lib first).
    """

    def __init__(self, data: torch.Tensor, context: int, batch_size: int, seed: int = 42):
        if context < 1 or batch_size < 1 or len(data) <= context:
            raise ValueError("Data must contain at least context + 1 bytes; batch_size must be positive")
        self.data = data
        self.context = context
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return max(1, (len(self.data) - self.context) // self.batch_size)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of (input, target) pairs."""
        starts = torch.randint(
            len(self.data) - self.context,
            (self.batch_size,),
            generator=self.generator,
        )
        tokens = torch.stack([self.data[i : i + self.context + 1] for i in starts])
        return tokens[:, :-1], tokens[:, 1:]


class Trainer:
    """Handles the full training lifecycle."""

    def __init__(
        self,
        model: MotanaxyModel,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        config: TrainingConfig,
        out_dir: str | Path,
    ):
        self.config = config
        self.device = config.resolve_device()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Model
        self.model = model.to(self.device)
        context = model.config["context"]

        # Data loaders
        self.train_loader = DataLoader(train_data, context, config.batch_size, config.seed)
        self.val_loader = DataLoader(val_data, context, config.batch_size, config.seed + 1)

        # Optimizer (Ponytail: AdamW with weight decay, no fancy lib)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # AMP scaler (only for CUDA)
        self.use_amp = config.use_amp and self.device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.amp_dtype = torch.float16 if self.use_amp else torch.float32

        # State
        self.step = 0
        self.best_val_loss = float("inf")
        self.history: list[dict] = []

    def get_lr(self) -> float:
        """Cosine schedule with linear warmup."""
        cfg = self.config
        if cfg.lr_schedule == "constant":
            return cfg.learning_rate

        # Linear warmup
        if self.step < cfg.warmup_steps:
            return cfg.learning_rate * (self.step + 1) / cfg.warmup_steps

        # Cosine decay
        progress = (self.step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        progress = min(progress, 1.0)
        min_lr = cfg.learning_rate * cfg.min_lr_ratio
        return min_lr + 0.5 * (cfg.learning_rate - min_lr) * (1 + math.cos(math.pi * progress))

    def _set_lr(self):
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        # Use identical windows so checkpoints are compared on the same examples.
        self.val_loader.generator.manual_seed(self.config.seed + 1)
        losses = []
        for _ in range(self.config.eval_batches):
            x, y = self.val_loader.sample()
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast(self.device, dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(loss.item())
        return sum(losses) / len(losses)

    def train_step(self) -> float:
        """Execute one training step (with gradient accumulation)."""
        self.model.train()
        total_loss = 0.0

        for micro_step in range(self.config.grad_accum_steps):
            x, y = self.train_loader.sample()
            x, y = x.to(self.device), y.to(self.device)

            with torch.amp.autocast(self.device, dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss = loss / self.config.grad_accum_steps

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; stopping before updating weights")

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item()

        # Gradient clipping
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        # Step
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        return total_loss

    def save_checkpoint(self, name: str = "checkpoint.pt"):
        """Save model + optimizer + training state."""
        path = self.out_dir / name
        temporary = path.with_suffix(".pt.tmp")
        torch.save({
            "config": self.model.config,
            "state": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "training_config": self.config.to_dict(),
        }, temporary)
        temporary.replace(path)
        logger.info("Saved checkpoint: %s (step %d)", path, self.step)

    def save_metrics(self):
        """Save training history as JSON."""
        path = self.out_dir / "metrics.json"
        path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")

    def train(self) -> dict:
        """Full training loop. Returns final report."""
        cfg = self.config
        torch.set_num_threads(cfg.num_threads)
        torch.manual_seed(cfg.seed)

        logger.info("Device: %s | AMP: %s", self.device, self.use_amp)
        logger.info("Parameters: %s", f"{self.model.num_parameters:,}")
        logger.info("Train bytes: %d | Val bytes: %d",
                     len(self.train_loader.data), len(self.val_loader.data))

        initial_val_loss = self.evaluate()
        logger.info("Initial val loss: %.4f", initial_val_loss)

        if initial_val_loss < self.best_val_loss:
            self.best_val_loss = initial_val_loss
            self.save_checkpoint("best_model.pt")

        started = time.perf_counter()

        for self.step in range(1, cfg.max_steps + 1):
            lr = self._set_lr()
            train_loss = self.train_step()

            # Logging
            if self.step == 1 or self.step % 50 == 0 or self.step == cfg.max_steps:
                print(
                    f"step={self.step}/{cfg.max_steps}  "
                    f"loss={train_loss:.4f}  lr={lr:.2e}",
                    flush=True,
                )

            # Evaluation
            if self.step % cfg.eval_interval == 0 or self.step == cfg.max_steps:
                val_loss = self.evaluate()
                record = {
                    "step": self.step,
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "lr": round(lr, 6),
                    "elapsed": round(time.perf_counter() - started, 1),
                }
                self.history.append(record)
                print(f"  -> val_loss={val_loss:.4f}", flush=True)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")

            # Periodic save
            if self.step % cfg.save_interval == 0:
                self.save_checkpoint(f"step_{self.step}.pt")

        # Final save
        elapsed = time.perf_counter() - started
        self.save_checkpoint("final_model.pt")
        self.save_metrics()

        final_val_loss = self.evaluate()

        report = {
            "model_preset": None,
            "parameters": self.model.num_parameters,
            "steps": cfg.max_steps,
            "batch_size": cfg.batch_size,
            "grad_accum_steps": cfg.grad_accum_steps,
            "effective_batch_size": cfg.batch_size * cfg.grad_accum_steps,
            "learning_rate": cfg.learning_rate,
            "device": self.device,
            "use_amp": self.use_amp,
            "torch_version": str(torch.__version__),
            "train_bytes": len(self.train_loader.data),
            "val_bytes": len(self.val_loader.data),
            "initial_val_loss": round(initial_val_loss, 4),
            "final_val_loss": round(final_val_loss, 4),
            "best_val_loss": round(self.best_val_loss, 4),
            "improvement": round(initial_val_loss - final_val_loss, 4),
            "seconds": round(elapsed, 2),
        }

        report_path = self.out_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("\n" + json.dumps(report, indent=2))
        return report


def prepare_data(
    corpus: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split corpus into train/val/test tensors by document boundaries.

    Documents are split by triple-newline, then allocated to each split.
    This ensures no data leakage across splits.
    """
    docs = [d for d in corpus.split("\n\n\n") if d.strip()]
    if len(docs) < 5:
        raise ValueError(
            f"Need at least 5 documents for train/val/test split, got {len(docs)}"
        )

    n = len(docs)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val - n_test

    if n_train < 3:
        raise ValueError(f"Not enough documents for training: {n_train}")

    test_docs = docs[:n_test]
    val_docs = docs[n_test : n_test + n_val]
    train_docs = docs[n_test + n_val :]

    def to_tensor(doc_list):
        text = "\n\n".join(doc_list)
        return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)

    return to_tensor(train_docs), to_tensor(val_docs), to_tensor(test_docs)
