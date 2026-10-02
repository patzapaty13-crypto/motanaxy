"""Verify AI-authored lessons, adapt the existing model, compare frozen holdouts."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from motanaxy.model import load_checkpoint
from motanaxy.trainer import DataLoader, Trainer, TrainingConfig
from training_data import verified_lessons as lessons


def curriculum():
    checks = lessons.check()
    names = sorted(lessons.CASES)
    random.Random(2026).shuffle(names)
    # Split by function BEFORE expanding translations or examples.
    groups = {'test': names[:6], 'validation': names[6:12], 'train': names[12:]}
    records = {}
    for split, members in groups.items():
        records[split] = []
        for name in members:
            function = getattr(lessons, name)
            source = inspect.getsource(function)
            tree = ast.parse(source)
            node = tree.body[0]
            english, thai = inspect.getdoc(function).split(' | ')
            # Keep task descriptions in comments, matching the existing chat prompt.
            del node.body[0]
            code = ast.unparse(tree) + '\n'
            examples = []
            for arguments, expected in lessons.CASES[name]:
                arguments_text = ', '.join(repr(arg) for arg in arguments)
                examples.append(f'assert {name}({arguments_text}) == {expected!r}')
            for language, description in [('en', english), ('th', thai)]:
                document = f'# {description}\n{code}\n' + '\n'.join(examples)
                # Parse rather than execute generated documents. References already tested.
                ast.parse(document)
                records[split].append(dict(concept=name, language=language,
                                           prompt=f'# {description}\n', code=code,
                                           text=document))
    return groups, records, checks


def tensor(records):
    text = '\n\n\n'.join(record['text'] for record in records)
    return torch.tensor(list(text.encode('utf-8')), dtype=torch.long)


@torch.no_grad()
def assess(model, test_records):
    """No model-generated code is executed; syntax success is not correctness."""
    model.eval()
    data = tensor(test_records)
    loader = DataLoader(data, model.config['context'], 4, seed=912)
    losses = []
    for _ in range(8):
        x, y = loader.sample()
        losses.append(F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1)).item())
    samples = []
    for record in test_records:
        prompt = record['prompt']
        # Greedy (top_k=1): deterministic outputs before/after, no temperature luck.
        torch.manual_seed(123)
        output = bytes(model.generate(list(prompt.encode()), max_tokens=180,
                                      temperature=1.0, top_k=1)).decode('utf-8', errors='replace')
        continuation = output[len(prompt):]
        # Evaluate a completed first block when the model emits a blank line boundary.
        candidate = continuation.split('\n\n', 1)[0].strip()
        valid = False
        try:
            tree = ast.parse(candidate)
            valid = bool(tree.body) and isinstance(tree.body[0], ast.FunctionDef)
        except (SyntaxError, ValueError):
            pass
        samples.append(dict(concept=record['concept'], language=record['language'],
                            prompt=prompt, output=continuation,
                            candidate=candidate, parses_as_function=valid))
    return dict(loss=sum(losses) / len(losses),
                syntactically_valid_functions=sum(s['parses_as_function'] for s in samples),
                prompts=len(samples), samples=samples,
                limitation='Syntax only; no generated-code execution or correctness claim. '
                           'Holdout excludes new training concepts, but prior checkpoint data may overlap.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', default='runs/mtn_full/best_model.pt')
    parser.add_argument('--out', default='runs/knowledge_v1')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    groups, records, checks = curriculum()
    print(f'PASS: {len(lessons.CASES)} concepts; {checks} reference checks', flush=True)
    all_groups = [set(value) for value in groups.values()]
    assert sum(map(len, all_groups)) == len(set.union(*all_groups)), 'concept leakage'
    if args.check:
        # Minimum legal batch length must sample the last/only possible window.
        x, y = DataLoader(torch.arange(9), 8, 1).sample()
        assert x.tolist() == [list(range(8))] and y.tolist() == [list(range(1, 9))]
        from tempfile import TemporaryDirectory
        from motanaxy.model import MotanaxyModel
        with TemporaryDirectory() as directory:
            tiny = MotanaxyModel(width=16, context=8, layers=1, heads=2)
            values = torch.arange(40)
            trainer = Trainer(tiny, values, values, TrainingConfig(
                batch_size=1, eval_batches=2, device='cpu', use_amp=False), directory)
            assert trainer.evaluate() == trainer.evaluate(), 'validation changed between comparisons'
            trainer.save_checkpoint()
            restored = load_checkpoint(str(Path(directory) / 'checkpoint.pt'))
            assert torch.equal(tiny(values[:8][None]), restored(values[:8][None]))
            def nonfinite_forward(inputs):
                return torch.full((*inputs.shape, 256), float('nan'))
            tiny.forward = nonfinite_forward
            try:
                trainer.train_step()
            except RuntimeError as exc:
                assert 'Non-finite' in str(exc)
            else:
                raise AssertionError('Non-finite loss was not rejected')
        print('PASS: disjoint splits, syntax, sampling, fixed validation, checkpoint reload, non-finite guard')
        return
    out = Path(args.out)
    if out.exists():
        parser.error('Output exists; choose a new --out to preserve earlier experiments')
    parent = Path(args.parent)
    model = load_checkpoint(str(parent))
    out.mkdir(parents=True)
    for split, entries in records.items():
        (out / f'{split}.jsonl').write_text(
            '\n'.join(json.dumps(row, ensure_ascii=False) for row in entries) + '\n', encoding='utf-8')
    manifest = dict(source='AI-authored local verified_lessons.py; no downloaded datasets or API calls',
                    knowledge_scope='60 Python concepts, not the assistant entire knowledge or weights',
                    reference_checks=checks, groups=groups,
                    documents={split: len(rows) for split, rows in records.items()},
                    source_sha256=hashlib.sha256(Path(lessons.__file__).read_bytes()).hexdigest(),
                    parent=str(parent.resolve()), parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
                    split_seed=2026, training_seed=42)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    before = assess(model, records['test'])
    (out / 'baseline.json').write_text(json.dumps(before, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Baseline heldout loss={before["loss"]:.4f}; syntax={before["syntactically_valid_functions"]}/{before["prompts"]}', flush=True)
    config = TrainingConfig(max_steps=args.steps, learning_rate=2e-4, warmup_steps=50,
                            eval_interval=250, save_interval=args.steps,
                            eval_batches=8, device='cpu', use_amp=False)
    trainer = Trainer(model, tensor(records['train']), tensor(records['validation']), config, out)
    assert trainer.evaluate() == trainer.evaluate(), 'validation windows must stay fixed'
    trainer.train()
    best = load_checkpoint(str(out / 'best_model.pt'))
    after = assess(best, records['test'])
    comparison = dict(before=before, after=after,
                      best_step=torch.load(out / 'best_model.pt', weights_only=True)['step'],
                      heldout_loss_improved=after['loss'] < before['loss'],
                      note='One bounded experiment. Does not establish general coding competence.')
    (out / 'comparison.json').write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Final heldout loss={after["loss"]:.4f}; syntax={after["syntactically_valid_functions"]}/{after["prompts"]}', flush=True)
    print(f'Results: {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
