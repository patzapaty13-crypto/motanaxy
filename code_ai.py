"""Small byte-level causal Transformer, trained from random weights. No API."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F


class CodeModel(nn.Module):
    def __init__(self, width=64, context=128, layers=2):
        super().__init__()
        self.config = dict(width=width, context=context, layers=layers)
        self.tokens = nn.Embedding(256, width)
        self.positions = nn.Embedding(context, width)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0,
                                       batch_first=True, activation="gelu")
            for _ in range(layers)
        ])
        self.head = nn.Linear(width, 256)

    def forward(self, x):
        n = x.shape[1]
        x = self.tokens(x) + self.positions(torch.arange(n, device=x.device))
        mask = torch.ones(n, n, dtype=torch.bool, device=x.device).triu(1)
        for block in self.blocks:
            x = block(x, src_mask=mask)
        return self.head(x)


def batch(data, context, batch_size, generator):
    starts = torch.randint(len(data) - context, (batch_size,), generator=generator)
    tokens = torch.stack([data[i:i + context + 1] for i in starts])
    return tokens[:, :-1], tokens[:, 1:]


@torch.no_grad()
def measure(model, data, context):
    model.eval()
    generator = torch.Generator().manual_seed(123)
    values = []
    for _ in range(8):
        x, y = batch(data, context, 8, generator)
        values.append(F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1)).item())
    return sum(values) / len(values)


def train(args):
    if args.steps < 1:
        raise ValueError("steps must be positive")
    out = Path(args.out)
    if out.exists():
        raise ValueError("Output already exists; choose a new --out to preserve earlier runs")
    raw = Path(args.data).read_bytes()
    # ponytail: whole corpus in RAM; use streamed document shards for large datasets.
    docs = [part for part in raw.split(b"\n\n\n") if part.strip()]
    if len(docs) < 4:
        raise ValueError("Need at least 4 documents separated by three newlines")
    # Split documents before tokenization so windows cannot cross the train/val split.
    cut = max(1, len(docs) // 5)
    val = torch.tensor(list(b"\n\n".join(docs[:cut])), dtype=torch.long)
    data = torch.tensor(list(b"\n\n".join(docs[cut:])), dtype=torch.long)
    model = CodeModel()
    context = model.config['context']
    if min(len(val), len(data)) <= context:
        raise ValueError("Each split needs more than 128 UTF-8 bytes")
    before = measure(model, val, context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    generator = torch.Generator().manual_seed(42)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        x, y = batch(data, context, 8, generator)
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss; training aborted")
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            print(f"step={step} train_loss={loss.item():.4f}", flush=True)
    after = measure(model, val, context)
    report = dict(parameters=sum(p.numel() for p in model.parameters()),
                  steps=args.steps, seed=42, device="cpu", torch_version=str(torch.__version__),
                  train_bytes=len(data), validation_bytes=len(val),
                  data_sha256=hashlib.sha256(raw).hexdigest(),
                  initial_validation_loss=before, final_validation_loss=after,
                  seconds=round(time.perf_counter() - started, 2),
                  limitation="Tiny synthetic corpus; loss is not a coding benchmark")
    out.mkdir(parents=True)
    torch.save(dict(config=model.config, state=model.state_dict()), out / "model.pt")
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


@torch.no_grad()
def generate(args):
    if not args.prompt or not 1 <= args.tokens <= 2048:
        raise ValueError("Provide a prompt and 1..2048 tokens")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = CodeModel(**checkpoint['config'])
    model.load_state_dict(checkpoint['state'])
    model.eval()
    ids = list(args.prompt.encode('utf-8'))
    for _ in range(args.tokens):
        x = torch.tensor([ids[-model.config['context']:]])
        logits = model(x)[0, -1] / args.temperature
        ids.append(torch.multinomial(logits.softmax(-1), 1).item())
    print(bytes(ids).decode('utf-8', errors='replace'))


def self_check():
    model = CodeModel(width=16, context=16, layers=1).eval()
    x = torch.randint(256, (2, 12))
    other = x.clone()
    other[:, 6:] = torch.randint(256, (2, 6))
    assert torch.allclose(model(x)[:, :6], model(other)[:, :6], atol=1e-6), 'Future token leakage'
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    target = (x + 1) % 256
    first = F.cross_entropy(model(x).reshape(-1, 256), target.reshape(-1)).item()
    for _ in range(30):
        loss = F.cross_entropy(model(x).reshape(-1, 256), target.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < first / 2, 'Learning failed'
    clone = CodeModel(**model.config)
    clone.load_state_dict(model.state_dict())
    assert torch.allclose(model(x), clone(x), atol=1e-6)
    sample = 'def hello(): # สวัสดี'
    assert bytes(list(sample.encode())).decode() == sample
    print('PASS: causal mask, gradient learning, state reload, UTF-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', default=str(Path(__file__).with_name('sample_data.txt')))
    t.add_argument('--steps', type=int, default=300)
    t.add_argument('--out', default='runs/demo')
    g = sub.add_parser('generate')
    g.add_argument('--checkpoint', default='runs/demo/model.pt')
    g.add_argument('--prompt', default='def ')
    g.add_argument('--tokens', type=int, default=160)
    g.add_argument('--temperature', type=float, default=0.7)
    sub.add_parser('check')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    try:
        {'train': train, 'generate': generate, 'check': lambda _: self_check()}[args.command](args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
