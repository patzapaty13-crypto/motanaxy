"""MOTANAXY Auto-Evolution Script for Kaggle/Colab.

This script runs FULLY AUTONOMOUSLY. Just upload and press Play.
It will:
1. Fix the tokenizer (proper vocab size, no over-compression)
2. Generate diverse training data (10k+ examples)
3. Train the model for multiple rounds
4. Evaluate and save the best version
5. Print a full report at the end

Designed to run on Kaggle (12hr GPU session) or Colab (T4 GPU).
No human intervention needed after pressing Play.
"""

import ast
import json
import os
import random
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Auto-detect GPU
# ---------------------------------------------------------------------------
import torch

if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("No GPU found, using CPU (will be slower)")

torch.manual_seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Step 0: Install dependencies if needed (for Kaggle/Colab)
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("MOTANAXY Auto-Evolution Pipeline")
print("="*60)

# ---------------------------------------------------------------------------
# Step 1: Generate DIVERSE Dataset (fixing over-compression issue)
# ---------------------------------------------------------------------------
print("\n[STEP 1/5] Generating diverse training data...")

def random_name(prefix="func"):
    vowels = "aeiou"
    consonants = "bcdfghjklmnprstvwxyz"
    name = prefix + "_" + "".join(
        random.choice(consonants) + random.choice(vowels) for _ in range(2)
    )
    return name


# --- Generator Category 1: Math Operations ---
def gen_math():
    ops = [
        ("a + b", "add", 2), ("a - b", "sub", 2), ("a * b", "mul", 2),
        ("a // b", "div", 2), ("a ** b", "pow", 2), ("a % b", "mod", 2),
        ("a + b + c", "add3", 3), ("(a + b) * c", "math1", 3),
        ("a * b - c", "math2", 3), ("abs(a - b)", "diff", 2),
        ("max(a, b)", "bigger", 2), ("min(a, b)", "smaller", 2),
        ("a * a + b * b", "sumsq", 2), ("(a + b) // 2", "avg", 2),
    ]
    expr, prefix, nargs = random.choice(ops)
    name = random_name(prefix)
    params = ", ".join("abcdefg"[:nargs])
    code = f"def {name}({params}):\n    return {expr}\n"
    
    def make_args():
        if "**" in expr:
            return tuple(random.randint(1, 5) for _ in range(nargs))
        elif "//" in expr or "%" in expr:
            args = [random.randint(1, 20) for _ in range(nargs)]
            args[1] = max(args[1], 1)  # avoid division by zero
            return tuple(args)
        return tuple(random.randint(1, 20) for _ in range(nargs))
    
    return name, code, f"Compute {expr}.", make_args


# --- Generator Category 2: List Operations ---
def gen_list():
    ops = [
        ("return max(lst)", "find_max", lambda: [random.randint(1,100) for _ in range(random.randint(3,7))]),
        ("return min(lst)", "find_min", lambda: [random.randint(1,100) for _ in range(random.randint(3,7))]),
        ("return sum(lst)", "total", lambda: [random.randint(1,20) for _ in range(random.randint(2,6))]),
        ("return len(lst)", "count", lambda: [random.randint(1,10) for _ in range(random.randint(1,8))]),
        ("return lst[::-1]", "flip", lambda: [random.randint(1,50) for _ in range(random.randint(2,5))]),
        ("return sorted(lst)", "order", lambda: [random.randint(1,50) for _ in range(random.randint(2,6))]),
        ("return lst[0] if lst else None", "head", lambda: [random.randint(1,10)] * random.randint(1,3)),
        ("return lst[-1] if lst else None", "tail", lambda: [random.randint(1,10)] * random.randint(1,3)),
        ("return [x * 2 for x in lst]", "double", lambda: [random.randint(1,20) for _ in range(random.randint(2,5))]),
        ("return [x for x in lst if x > 0]", "positives", lambda: [random.randint(-10,10) for _ in range(random.randint(3,6))]),
        ("return list(set(lst))", "unique", lambda: [random.choice([1,2,3,4,5]) for _ in range(random.randint(3,7))]),
        ("return sum(lst) / len(lst) if lst else 0", "average", lambda: [random.randint(1,20) for _ in range(random.randint(2,5))]),
    ]
    body, prefix, arg_gen = random.choice(ops)
    name = random_name(prefix)
    code = f"def {name}(lst):\n    {body}\n"
    return name, code, f"List operation: {prefix.replace('_',' ')}.", lambda: (arg_gen(),)


# --- Generator Category 3: String Operations ---
def gen_string():
    ops = [
        ("return len(s)", "str_len", lambda: random.choice(["hello", "world", "python", "code", "test", "abc", "xyz123"])),
        ("return s.upper()", "to_upper", lambda: random.choice(["hello", "world", "python", "code"])),
        ("return s.lower()", "to_lower", lambda: random.choice(["HELLO", "WORLD", "PYTHON", "CODE"])),
        ("return s[::-1]", "str_flip", lambda: random.choice(["hello", "world", "abcde", "12345"])),
        ("return s.count(c)", "char_count", None),
        ("return s.startswith(c)", "starts", None),
        ("return s.replace(old, new)", "swap", None),
    ]
    body, prefix, simple_gen = random.choice(ops[:4])  # use only simple ones
    name = random_name(prefix)
    code = f"def {name}(s):\n    {body}\n"
    return name, code, f"String operation: {prefix.replace('_',' ')}.", lambda: (simple_gen(),)


# --- Generator Category 4: Conditional Logic ---
def gen_conditional():
    templates = [
        ("def {name}(x):\n    if x > 0:\n        return 'positive'\n    elif x < 0:\n        return 'negative'\n    return 'zero'\n",
         "sign", lambda: (random.randint(-10, 10),)),
        ("def {name}(x):\n    if x % 2 == 0:\n        return 'even'\n    return 'odd'\n",
         "parity", lambda: (random.randint(1, 100),)),
        ("def {name}(a, b):\n    if a > b:\n        return a\n    return b\n",
         "pick_big", lambda: (random.randint(1, 50), random.randint(1, 50))),
        ("def {name}(a, b):\n    if a < b:\n        return a\n    return b\n",
         "pick_small", lambda: (random.randint(1, 50), random.randint(1, 50))),
        ("def {name}(x):\n    if x >= 0:\n        return x\n    return -x\n",
         "magnitude", lambda: (random.randint(-20, 20),)),
        ("def {name}(lst):\n    count = 0\n    for x in lst:\n        if x > 0:\n            count += 1\n    return count\n",
         "count_pos", lambda: ([random.randint(-10, 10) for _ in range(random.randint(3,6))],)),
    ]
    template, prefix, args_gen = random.choice(templates)
    name = random_name(prefix)
    code = template.format(name=name)
    return name, code, f"Conditional logic: {prefix.replace('_',' ')}.", args_gen


# --- Generator Category 5: Loop-based algorithms ---
def gen_loop():
    templates = [
        ("def {name}(n):\n    total = 0\n    for i in range(1, n + 1):\n        total += i\n    return total\n",
         "sum_to", lambda: (random.randint(1, 20),)),
        ("def {name}(n):\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result\n",
         "factorial", lambda: (random.randint(1, 8),)),
        ("def {name}(lst):\n    total = 0\n    for x in lst:\n        total += x\n    return total\n",
         "manual_sum", lambda: ([random.randint(1, 10) for _ in range(random.randint(2, 5))],)),
        ("def {name}(lst):\n    biggest = lst[0]\n    for x in lst[1:]:\n        if x > biggest:\n            biggest = x\n    return biggest\n",
         "manual_max", lambda: ([random.randint(1, 100) for _ in range(random.randint(2, 6))],)),
        ("def {name}(n):\n    result = []\n    for i in range(n):\n        result.append(i * i)\n    return result\n",
         "squares", lambda: (random.randint(1, 8),)),
    ]
    template, prefix, args_gen = random.choice(templates)
    name = random_name(prefix)
    code = template.format(name=name)
    return name, code, f"Loop algorithm: {prefix.replace('_',' ')}.", args_gen


def build_example(gen_func):
    """Build a verified prompt/code/test example."""
    name, code, doc, args_gen = gen_func()
    
    env = {}
    try:
        exec(code, {}, env)
    except Exception:
        return None
    
    func = env[name]
    tests = []
    for _ in range(3):
        args = args_gen()
        try:
            result = func(*args)
            args_str = ", ".join(repr(a) for a in args)
            tests.append(f"assert {name}({args_str}) == {result!r}")
        except Exception:
            continue
    
    if not tests:
        return None
    
    prompt = f"# {doc}\n"
    full = prompt + code + "\n" + "\n".join(tests) + "\n"
    return {"concept": name, "prompt": prompt, "code": code, "text": full}


def generate_dataset(n=10000):
    """Generate n diverse verified examples."""
    generators = [gen_math, gen_list, gen_string, gen_conditional, gen_loop]
    weights = [0.25, 0.25, 0.15, 0.2, 0.15]
    
    results = []
    attempts = 0
    while len(results) < n and attempts < n * 3:
        attempts += 1
        gen = random.choices(generators, weights=weights, k=1)[0]
        ex = build_example(gen)
        if ex:
            results.append(ex)
        if len(results) % 2000 == 0 and len(results) > 0:
            print(f"  Generated {len(results)}/{n} examples...")
    
    return results


# Generate the dataset
data = generate_dataset(12000)
print(f"  Total verified examples: {len(data)}")

# Split: 10k train, 1k val, 1k test
random.shuffle(data)
train_data = data[:10000]
val_data = data[10000:11000]
test_data = data[11000:]

# Save to files
os.makedirs("training_data", exist_ok=True)
with open("training_data/synthetic_diverse.jsonl", "w") as f:
    for d in train_data + val_data:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print(f"  Saved {len(train_data)} train + {len(val_data)} val examples")

# ---------------------------------------------------------------------------
# Step 2: Train a PROPER BPE Tokenizer (with size limit to prevent 
# over-compression)
# ---------------------------------------------------------------------------
print("\n[STEP 2/5] Training BPE Tokenizer (vocab=1024, prevents over-compression)...")

# We use a SMALL vocab (1024) to prevent the tokenizer from memorizing
# entire functions as single tokens. This forces the model to actually
# learn the structure character-by-character at a reasonable granularity.

from motanaxy.tokenizer import BPETokenizer

# Combine all training text
all_text = "\n\n".join(d["text"] for d in random.sample(train_data, min(2000, len(train_data))))

tokenizer = BPETokenizer()
tokenizer.train(all_text, vocab_size=1024, verbose=True)
tokenizer.save("motanaxy/tokenizer_v2.json")

# Verify compression ratio
test_snippet = "def add_xy(a, b):\n    return a + b\n\nassert add_xy(3, 5) == 8\n"
encoded = tokenizer.encode(test_snippet)
ratio = len(test_snippet.encode()) / len(encoded)
print(f"  Compression ratio: {ratio:.2f}x (target: 2-4x, NOT 6x+)")
print(f"  Vocab size: {len(tokenizer.vocab)}")

if ratio > 5:
    print("  WARNING: Compression too high, reducing vocab...")
    tokenizer = BPETokenizer()
    tokenizer.train(all_text, vocab_size=512, verbose=False)
    tokenizer.save("motanaxy/tokenizer_v2.json")
    encoded = tokenizer.encode(test_snippet)
    ratio = len(test_snippet.encode()) / len(encoded)
    print(f"  New compression ratio: {ratio:.2f}x")

# ---------------------------------------------------------------------------
# Step 3: Tokenize the dataset
# ---------------------------------------------------------------------------
print("\n[STEP 3/5] Tokenizing dataset...")

tokenizer_v2 = BPETokenizer()
tokenizer_v2.load("motanaxy/tokenizer_v2.json")
vocab_size = len(tokenizer_v2.vocab)

def tokenize_records(records):
    tokens = []
    for r in records:
        tokens.extend(tokenizer_v2.encode(r["text"] + "\n\n"))
    return torch.tensor(tokens, dtype=torch.long)

train_tokens = tokenize_records(train_data)
val_tokens = tokenize_records(val_data)
print(f"  Training tokens: {len(train_tokens):,}")
print(f"  Validation tokens: {len(val_tokens):,}")
print(f"  Vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# Step 4: Build and Train Model (multiple rounds with auto-tuning)
# ---------------------------------------------------------------------------
print("\n[STEP 4/5] Training model (auto-evolution mode)...")

from motanaxy.model import MotanaxyModel
from motanaxy.trainer import DataLoader, Trainer, TrainingConfig

# Model config: ~10M params with proper vocab
model_config = dict(
    width=256, context=256, layers=12, heads=8,
    version=2, dropout=0.1, vocab_size=vocab_size
)

model = MotanaxyModel(**model_config)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {n_params:,}")
print(f"  Config: {model_config}")
print(f"  Device: {DEVICE}")

# Training rounds with different strategies
ROUNDS = [
    {"steps": 2000, "lr": 3e-4, "warmup": 300, "batch": 8, "accum": 2, "label": "Round 1: Initial Learning"},
    {"steps": 2000, "lr": 1e-4, "warmup": 100, "batch": 8, "accum": 2, "label": "Round 2: Fine-tuning"},
    {"steps": 1000, "lr": 3e-5, "warmup": 50,  "batch": 8, "accum": 2, "label": "Round 3: Polish"},
]

best_val_loss = float("inf")
round_results = []
total_start = time.perf_counter()

for round_idx, rnd in enumerate(ROUNDS):
    print(f"\n--- {rnd['label']} ({rnd['steps']} steps, lr={rnd['lr']}) ---")
    
    out_dir = Path(f"runs/auto_round_{round_idx + 1}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    config = TrainingConfig(
        max_steps=rnd["steps"],
        learning_rate=rnd["lr"],
        warmup_steps=rnd["warmup"],
        batch_size=rnd["batch"],
        grad_accum_steps=rnd["accum"],
        eval_interval=250,
        eval_batches=16,
        save_interval=500,
        device=DEVICE,
        use_amp=(DEVICE == "cuda"),
        weight_decay=0.1,
    )
    
    trainer = Trainer(model, train_tokens, val_tokens, config, out_dir)
    trainer.train()
    
    # Check if this round improved
    current_val = trainer.best_val_loss if hasattr(trainer, 'best_val_loss') else float("inf")
    
    # Also try loading best checkpoint's val loss from the training summary
    summary_path = out_dir / "training_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        current_val = summary.get("best_val_loss", current_val)
    
    improved = current_val < best_val_loss
    if improved:
        best_val_loss = current_val
        # Copy best model
        best_ckpt = out_dir / "best_model.pt"
        if best_ckpt.exists():
            torch.save(torch.load(best_ckpt, weights_only=True), "runs/best_overall.pt")
    
    round_results.append({
        "round": round_idx + 1,
        "label": rnd["label"],
        "steps": rnd["steps"],
        "lr": rnd["lr"],
        "val_loss": current_val,
        "improved": improved,
    })
    
    print(f"  Val loss: {current_val:.4f} {'(NEW BEST!)' if improved else ''}")

total_time = time.perf_counter() - total_start

# ---------------------------------------------------------------------------
# Step 5: Final Assessment
# ---------------------------------------------------------------------------
print("\n[STEP 5/5] Final assessment on held-out test set...")

model.eval()

@torch.no_grad()
def assess_model(model, test_records, n=20):
    results = []
    for rec in test_records[:n]:
        prompt = rec["prompt"]
        prompt_ids = tokenizer_v2.encode(prompt)
        
        output_ids = model.generate(prompt_ids, max_tokens=150, temperature=0.7, top_k=40)
        output_text = tokenizer_v2.decode(output_ids)
        continuation = output_text[len(prompt):]
        
        # Extract just the function (stop at double newline or next comment)
        candidate = continuation.split("\n\n")[0].strip()
        
        # Check syntax validity
        valid_syntax = False
        try:
            tree = ast.parse(candidate)
            valid_syntax = bool(tree.body) and isinstance(tree.body[0], ast.FunctionDef)
        except (SyntaxError, ValueError):
            pass
        
        # Check if the function actually runs correctly
        logic_correct = False
        if valid_syntax:
            try:
                # Try executing the generated code + its asserts
                full_code = candidate
                # Find assert lines in the original test data
                original_asserts = [l for l in rec["text"].split("\n") if l.startswith("assert")]
                if original_asserts:
                    # Replace function name in asserts with generated function name
                    gen_tree = ast.parse(candidate)
                    gen_name = gen_tree.body[0].name
                    test_code = full_code + "\n"
                    for a in original_asserts:
                        test_code += a.replace(rec["concept"], gen_name) + "\n"
                    exec(test_code, {}, {})
                    logic_correct = True
            except Exception:
                pass
        
        results.append({
            "concept": rec["concept"],
            "prompt": prompt,
            "output": continuation[:200],
            "candidate": candidate[:200],
            "syntax_valid": valid_syntax,
            "logic_correct": logic_correct,
        })
    
    syntax_score = sum(r["syntax_valid"] for r in results)
    logic_score = sum(r["logic_correct"] for r in results)
    return results, syntax_score, logic_score

samples, syntax_score, logic_score = assess_model(model, test_data)
print(f"  Syntax accuracy: {syntax_score}/{len(samples)}")
print(f"  Logic accuracy:  {logic_score}/{len(samples)}")

# ---------------------------------------------------------------------------
# Final Report
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("FINAL REPORT")
print("="*60)

report = {
    "model": {
        "parameters": n_params,
        "config": model_config,
        "architecture": "v2 (RMSNorm + RoPE + SwiGLU + Dropout)",
    },
    "dataset": {
        "total_examples": len(data),
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data),
        "categories": ["math", "list", "string", "conditional", "loop"],
    },
    "tokenizer": {
        "vocab_size": vocab_size,
        "compression_ratio": ratio,
    },
    "training": {
        "total_steps": sum(r["steps"] for r in ROUNDS),
        "total_time_seconds": total_time,
        "total_time_hours": total_time / 3600,
        "device": DEVICE,
        "rounds": round_results,
        "best_val_loss": best_val_loss,
    },
    "evaluation": {
        "syntax_score": f"{syntax_score}/{len(samples)}",
        "logic_score": f"{logic_score}/{len(samples)}",
        "samples": samples[:5],  # save first 5 for inspection
    },
}

# Save report
os.makedirs("runs", exist_ok=True)
with open("runs/auto_evolution_report.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)

print(f"\nModel:     {n_params:,} parameters")
print(f"Dataset:   {len(train_data)} train / {len(val_data)} val / {len(test_data)} test")
print(f"Tokenizer: vocab={vocab_size}, compression={ratio:.2f}x")
print(f"Time:      {total_time:.0f}s ({total_time/3600:.2f} hours)")
print(f"Best Loss: {best_val_loss:.4f}")
print(f"Syntax:    {syntax_score}/{len(samples)}")
print(f"Logic:     {logic_score}/{len(samples)}")
print(f"\nFull report saved to: runs/auto_evolution_report.json")
print(f"Best model saved to:  runs/best_overall.pt")
print("\n" + "="*60)
print("AUTO-EVOLUTION COMPLETE")
print("="*60)
