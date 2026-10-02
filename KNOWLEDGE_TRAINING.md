# MOTANAXY: verified Python curriculum

This experiment transfers a finite set of AI-authored examples, not the assistant's internal weights or all of its knowledge. It continues the existing 858,880-parameter model in `runs/mtn_full/best_model.pt`, using only local CPU computation. It does not purchase API, cloud, or GPU time. Electricity, compute time, and disk usage still apply.

## Dataset and checks

`training_data/verified_lessons.py` contains 60 reference functions with Thai and English task descriptions, covering basic mathematics, strings, sequences, sorting/searching, dictionaries, graphs, JSON/CSV, URL encoding, checksums, dates, elementary numerical operations, and parameterized SQL. Its 141 executable checks verify reference examples, including selected error cases. These are not tests passed by the model.

The curriculum reserves 6 concepts for validation and 6 for final testing. The remaining 48 concepts produce 96 bilingual training documents. Translations and examples of a function always stay together. The held-out concepts are excluded from this training run, but similar concepts may already exist in the parent model's old corpus; this is not a contamination-free public benchmark.

The checkpoint's 256-byte context and small capacity remain unchanged. Training these examples cannot establish general conversation ability, reliable Thai responses, or reliable programming competence. Generation is still probabilistic and may be nonsensical. Do not treat more steps or lower loss as proof of correctness.

## Reproduce

Run from `C:\AI project` in PowerShell:

```powershell
.\.venv\Scripts\python.exe train_knowledge.py --check
.\.venv\Scripts\python.exe train_knowledge.py --steps 2000 --out runs/knowledge_v2
```

The second command starts another CPU experiment using the original parent weights; it does not resume the previous optimizer. An existing output directory is rejected to preserve earlier experiments. To adapt different weights, supply `--parent path/to/best_model.pt`.

Results from this run are in `runs/knowledge_v1/`:

- `manifest.json`: provenance, source/parent hashes, exact concept split and document counts.
- `train.jsonl`, `validation.jsonl`, `test.jsonl`: inspectable corpus records.
- `baseline.json`: original model scores and literal generated outputs.
- `best_model.pt`: checkpoint selected using fixed validation windows; step zero is retained if training never improves validation.
- `final_model.pt`, `metrics.json`, `report.json`: final checkpoint and training evidence.
- `comparison.json`: original versus selected model on a fixed final test protocol, including literal outputs and the selected training step.

Comparison uses fixed test windows for next-byte cross-entropy and 12 bilingual prompts with greedy generation limited to 180 bytes. The first generated block is checked with `ast.parse` for a function definition. Generated code is never executed. A syntax pass may still return a wrong answer, and truncation can make a valid partial solution fail. This is a narrow smoke test, not pass@1 or a product benchmark.

## Use the checkpoint in the existing chat

Completed run: 2,000 CPU steps in 401.62 seconds. Selected checkpoint: step 500 (validation loss 1.8961). Frozen held-out loss improved from 2.6048 to 2.1128; syntax checks improved from 0/12 to 2/12. These are limited experimental results, not functional correctness scores. More training beyond step 500 increased overfitting on this curriculum.

The current frontend is Next.js at http://127.0.0.1:3000; see `frontend/README.md` for the two-server startup instructions. The original static page remains available at the backend URL for compatibility.

Restart the local app after training completes:

```powershell
.\.venv\Scripts\python.exe app_server.py
```

The app selects `knowledge_v1/best_model.pt` only if final held-out loss improves and syntax count does not decrease. Otherwise it keeps the old model. This automatic selection is a limited experiment gate, not a declaration that the model is useful. The sidebar reports the loaded run, parameters, and this run's training-document count. Counts are not cumulative across the parent model's training.

To explicitly select a checkpoint, set `MOTANAXY_CHECKPOINT` before starting the app. The old model and its reports remain intact. No external posts or fundraising announcements are published by this experiment.
