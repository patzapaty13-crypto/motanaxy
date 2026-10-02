# MOTANAXY — Token-Efficient Code Training System

**Byte-level Transformer** สำหรับเทรน code generation model จากข้อมูลจริง
ออกแบบตามหลัก **Ponytail** (เขียนน้อย ได้มาก) + **MarkItDown** (อ่านทุกรูปแบบไฟล์)

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      MOTANAXY Pipeline                       │
├──────────────┬───────────────┬──────────────┬───────────────┤
│   Ingestor   │    Model      │   Trainer    │   Generator   │
│              │               │              │               │
│ • Scan dirs  │ • Byte-level  │ • Cosine LR  │ • Top-k       │
│ • MarkItDown │ • Pre-norm    │ • Warmup     │ • Temperature │
│ • Dedup      │ • Weight tie  │ • Grad accum │ • Streaming   │
│ • Clean      │ • 4 presets   │ • AMP (GPU)  │               │
│ • Filter     │               │ • Checkpoint │               │
└──────────────┴───────────────┴──────────────┴───────────────┘
```

## 🚀 Quick Start

### 1. ติดตั้ง Dependencies

```bash
pip install torch "markitdown[all]"
```

### 2. Ingest ข้อมูล (รวบรวมไฟล์เป็น corpus)

```bash
# Ingest จากหลาย source
python -m motanaxy ingest --sources ./src ./docs ./examples --out corpus.txt

# Ingest ไฟล์เดียว
python -m motanaxy ingest --sources ./my_code.py --out corpus.txt

# ไม่ใช้ MarkItDown (เฉพาะ plaintext)
python -m motanaxy ingest --sources ./src --out corpus.txt --no-markitdown
```

### 3. Train Model

```bash
# ใช้ preset (แนะนำ)
python -m motanaxy train --corpus corpus.txt --preset nano --steps 500

# Custom config
python -m motanaxy train --corpus corpus.txt --width 128 --context 256 --layers 4 --steps 1000

# Resume training
python -m motanaxy train --corpus corpus.txt --preset micro --steps 2000 --resume --out runs/mtn
```

### 4. Generate Code

```bash
python -m motanaxy generate --checkpoint runs/mtn/best_model.pt --prompt "def sort_list(" --tokens 200
```

### 5. ดูข้อมูล Model

```bash
python -m motanaxy info --checkpoint runs/mtn/best_model.pt
```

### 6. Self-Check

```bash
python -m motanaxy check
```

---

## 📐 Model Presets

| Preset   | Width | Context | Layers | Heads | ~Params  | Use Case              |
|----------|-------|---------|--------|-------|----------|-----------------------|
| `nano`   | 64    | 128     | 2      | 4     | ~100K    | ทดสอบ / เรียนรู้       |
| `micro`  | 128   | 256     | 4      | 4     | ~800K    | ข้อมูลน้อย             |
| `small`  | 256   | 512     | 6      | 8     | ~5M      | ข้อมูลปานกลาง          |
| `medium` | 512   | 1024    | 8      | 8     | ~30M     | ข้อมูลมาก / GPU        |

---

## 🔧 Ponytail Principles Applied

1. **Lazy Loading** — MarkItDown โหลดเมื่อเจอไฟล์ที่ต้องใช้เท่านั้น
2. **No Duplicate Work** — Content hash deduplication
3. **Reuse Weights** — Weight tying ลด parameters 30%
4. **Standard Library First** — ใช้ `pathlib`, `hashlib`, `json` ก่อน library อื่น
5. **One Config Object** — `TrainingConfig` รวมทุก hyperparameter ไว้ที่เดียว

## 📁 Project Structure

```
motanaxy/
├── __init__.py      # Package metadata
├── __main__.py      # CLI entry point
├── ingestor.py      # Multi-format data ingestion
├── model.py         # Transformer architecture
└── trainer.py       # Training loop & utilities
```

---

## ⚡ Performance Tips

- **CPU**: ใช้ `nano` หรือ `micro` preset, batch_size=4-8
- **GPU**: ใช้ `small` หรือ `medium`, เปิด AMP อัตโนมัติ
- **Memory จำกัด**: เพิ่ม `--grad-accum 4` เพื่อลด memory ต่อ step
- **ข้อมูลน้อย**: ลด `--steps` เพิ่ม `--eval-interval` เพื่อจับ overfitting

---

*Built with ❤️ by MOTANAXY*
