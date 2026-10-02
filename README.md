# Motanaxy: A Tiny Language Model Built from Scratch 🧠

![Motanaxy Banner](https://img.shields.io/badge/Model-10M_Parameters-blue) ![Python](https://img.shields.io/badge/Built_with-PyTorch-red) ![Status](https://img.shields.io/badge/Status-Training-brightgreen)

## 🌟 Why build this? (ทำไปเพื่ออะไร?)

**[English]**  
In a world where everyone is simply calling APIs from massive companies (OpenAI, Anthropic, Google), the actual mechanics of how AI learns and reasons are becoming a "black box" to most developers. 

I started **Motanaxy** with one goal: **To truly understand how Language Models work under the hood.** 

Instead of fine-tuning a pre-existing multi-billion parameter model, I wanted to start with an absolutely empty brain (randomly initialized weights) and teach it to write Python code from scratch. This project is my journey of exploring tokenization limits, dataset procedural generation, attention mechanisms, and the threshold where a model stops "memorizing syntax" and begins "learning logic."

**[ภาษาไทย]**  
ในยุคที่ใครๆ ก็เขียนโปรแกรมด้วยการเรียกใช้ API จากบริษัทใหญ่ๆ (OpenAI, Anthropic, Google) กลไกการทำงานจริงๆ ว่า AI มันเรียนรู้และคิดได้ยังไงกลับกลายเป็น "กล่องดำ" (Black box) สำหรับนักพัฒนาส่วนใหญ่

ผมเริ่มโปรเจกต์ **Motanaxy** ด้วยเป้าหมายเดียวคือ: **ต้องการเข้าใจการทำงานเชิงลึกของ Language Model อย่างแท้จริง**

แทนที่จะเอาโมเดลตัวใหญ่ๆ ที่เก่งอยู่แล้วมาปรับแต่ง (Fine-tuning) ผมเลือกที่จะเริ่มต้นด้วย "สมองที่ว่างเปล่า" (สุ่มค่าน้ำหนักใหม่ทั้งหมด 100%) และสอนให้มันเขียนโค้ด Python ได้จากศูนย์ โปรเจกต์นี้คือการเดินทางเพื่อทดลองขีดจำกัดของการตัดคำ (Tokenization), การสร้างข้อมูลสอน (Dataset generation), กลไกความสนใจ (Attention) และค้นหาว่าจุดไหนที่ AI จะเลิก "จำแพทเทิร์น" แล้วหันมา "ใช้เหตุผล" จริงๆ

---

## 🎯 Project Goals (เป้าหมายของโปรเจกต์)

**[English]**
1. **Zero Pre-training:** Prove that a small model (~10M parameters) can learn Python syntax and simple algorithmic logic entirely from scratch.
2. **Local & Cloud Capable:** Build training scripts that can run on a local CPU or seamlessly transition to free cloud GPUs (like Kaggle or Colab) via `auto_evolve.py`.
3. **Custom Tokenizer:** Build a custom BPE tokenizer from the ground up to understand compression limits (and fix over-compression bugs!).
4. **Educational Open Source:** Provide a readable, minimal codebase for anyone who wants to learn how LLMs are constructed without the bloat of enterprise repositories.

**[ภาษาไทย]**
1. **ไม่พึ่งของสำเร็จรูป:** พิสูจน์ให้เห็นว่าโมเดลขนาดจิ๋ว (~10 ล้านพารามิเตอร์) สามารถเรียนรู้ไวยากรณ์ Python และตรรกะเบื้องต้นได้ด้วยตัวเอง 100%
2. **รันได้ทุกที่:** สร้างสคริปต์ที่รันได้ทั้งบน CPU กากๆ ที่บ้าน หรือโยนขึ้นไปเทรนบน GPU ฟรีบน Cloud (Kaggle/Colab) ได้อัตโนมัติผ่าน `auto_evolve.py`
3. **สร้าง Tokenizer เอง:** เขียนระบบตัดคำ (BPE) ขึ้นมาเองเพื่อศึกษาขีดจำกัดการบีบอัดข้อมูล (และแก้บั๊กการบีบอัดที่มากเกินไปจน AI ขี้โกง!)
4. **เป็นแหล่งเรียนรู้ Open Source:** เป็นโปรเจกต์ที่โค้ดอ่านง่าย ตรงไปตรงมา สำหรับคนที่อยากรู้ว่า LLM สร้างขึ้นมายังไง โดยไม่ต้องไปงมกับโค้ดระดับองค์กรที่ซับซ้อนเกินไป

---

## 🚀 Current Architecture (สถาปัตยกรรมปัจจุบัน)
- **Parameters:** ~10,000,000 (10M)
- **Layers:** 12
- **Heads:** 8
- **Embedding Width:** 256
- **Context Size:** 256 tokens
- **Architecture Features:** RMSNorm, RoPE (Rotary Positional Embeddings), SwiGLU activation.

## 🛠️ How to run (วิธีใช้งาน)

**Auto-Evolution on Cloud (Kaggle/Colab):**
```python
!unzip -o motanaxy_kaggle.zip
!python auto_evolve.py
```
*This will autonomously generate 12,000 diverse training examples, train a custom BPE tokenizer (vocab=1024), and train the model in 3 rounds.*

**Local Testing (Python environment):**
```bash
python generate_dataset.py
python train_xlarge.py --steps 2000
```
