# Code AI — ต้นแบบฝึกจากศูนย์

หน้าใช้งานปัจจุบันเป็น **MOTANAXY Next.js** อยู่ใน `frontend/` ดูวิธีรัน frontend + backend ใน `frontend/README.md` และรายละเอียดการฝึกต่อใน `KNOWLEDGE_TRAINING.md` ส่วนคำสั่ง `code_ai.py` ด้านล่างเป็นต้นแบบรุ่นแรกที่เก็บไว้

โมเดล Transformer ขนาดเล็ก เริ่มด้วยน้ำหนักสุ่ม ไม่มี pretrained weights และไม่เรียก API โมเดลอื่น ใช้ PyTorch คำนวณ gradient และใช้ UTF-8 bytes เป็น token จำนวน 256 ค่า

## เริ่มใช้งาน (PowerShell)

เปิด terminal ที่ `C:\AI project` สภาพแวดล้อม `.venv` ถูกเตรียมไว้แล้ว:

```powershell
.\.venv\Scripts\python.exe code_ai.py check
.\.venv\Scripts\python.exe code_ai.py train --steps 300 --out runs/experiment-01
.\.venv\Scripts\python.exe code_ai.py generate --checkpoint runs/experiment-01/model.pt --prompt "def add(" --tokens 160
```

สำหรับเครื่องใหม่ ติดตั้ง Python 3.12 แล้วสร้าง environment และติดตั้ง dependency:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

การฝึกจะบันทึก `model.pt` และ `metrics.json` ลงโฟลเดอร์ที่ระบุ หากโฟลเดอร์มีอยู่แล้วให้เปลี่ยนชื่อเพื่อรักษาผลเดิม โปรแกรมสร้างข้อความเท่านั้น ไม่รันโค้ดที่สร้างขึ้น ดาวน์โหลด checkpoint จากแหล่งที่เชื่อถือเท่านั้น

## ขอบเขตและหลักฐาน

ผลรันแรก: โมเดล 141,184 พารามิเตอร์ ฝึกบน CPU 300 steps เวลา training/evaluation ประมาณ 5.57 วินาที (ไม่รวมติดตั้งและโหลดไลบรารี) validation loss ลดจาก 5.7765 เป็น 2.6109 มี train 999 bytes และ validation 298 bytes เท่านั้น เก็บค่าจริงใน `runs/demo/metrics.json` และน้ำหนักใน `runs/demo/model.pt` ทดสอบ `check` ผ่าน และตรวจแล้วว่าคำสั่ง train ปฏิเสธการเขียนทับโฟลเดอร์ผลเดิม

ลองโมเดลที่ฝึกแล้วได้ทันที:

```powershell
.\.venv\Scripts\python.exe code_ai.py generate --prompt "def add(" --tokens 120
```

PyTorch อาจเตือนว่าไม่มี NumPy ใน environment นี้ โปรแกรมไม่ใช้ NumPy และการทดสอบกับการฝึกผ่านโดยไม่ต้องติดตั้งเพิ่ม

ผล generation จาก `def add(` โหลด checkpoint และสร้างข้อความได้จริง แต่ข้อความยังผิดไวยากรณ์ เช่น `def add(me):` ตามด้วย `reven ivaluers ...` จึงยังไม่ใช่ฟังก์ชัน Python ที่ใช้งานได้

- รองรับฝึก ทำนาย และบันทึก/โหลดน้ำหนักบน CPU ยังไม่มี chat, instruction tuning, IDE extension หรือ API server
- แบบจำลอง 2 ชั้น ความกว้าง 64, attention 4 heads, context 128 bytes
- ข้อมูลใน `sample_data.txt` เป็นตัวอย่างสังเคราะห์ที่ผู้ช่วย AI เขียนในงานนี้ ไม่ได้คัดลอก repository ภายนอก และไม่ใช่ชุดข้อมูลที่มนุษย์เขียนทั้งหมด
- แยกเอกสาร train/validation ก่อนสร้างหน้าต่างข้อมูล โดยคั่นเอกสารด้วย newline สามตัว ข้อมูลจริงต้องคัดข้อมูลซ้ำและข้อมูลใกล้เคียงข้ามชุดเพิ่มด้วย
- Loss ที่ลดลงพิสูจน์เพียงว่าระบบฝึกทำงาน ไม่ได้พิสูจน์ว่าแก้โจทย์ใหม่ได้ โค้ดที่สร้างอาจใช้ไม่ได้ทั้งหมด
- `check` ตรวจ causal attention ไม่เห็นอนาคต, การเรียนรู้ด้วย gradient, การโหลด state และ UTF-8
- ยังไม่มีผล coding benchmark, security benchmark หรือหลักฐานพร้อมขาย

คำว่า “ฝึกจากศูนย์” หมายถึงเริ่มน้ำหนักสุ่ม การใช้ข้อมูลสังเคราะห์จาก AI ไม่ได้ทำให้กลายเป็นการ fine-tune pretrained model โดยอัตโนมัติ หากใช้โมเดลครูผลิตชุดฝึกในอนาคตต้องบันทึกที่มาและตรวจสิทธิ์ของผู้ให้บริการ

## ต้นทุนและขั้นต่อไป

ต้นแบบใช้เครื่องเดิมและไม่ซื้อ cloud/API แต่ยังใช้ไฟฟ้า พื้นที่ดิสก์ และเวลาประมวลผล รอบเริ่มต้นจำกัดเพียง 300 steps เพื่อไม่ปล่อยงานหนักโดยไม่จำเป็น

ก่อนขยาย: รวบรวมข้อมูลที่มีสิทธิ์พร้อม provenance → แยกโจทย์ทดสอบที่ไม่อยู่ในข้อมูลฝึก → วัดการเติมฟังก์ชันด้วย unit tests ใน sandbox → จึงพิจารณาฝึกด้วย GPU และเพิ่มขนาดโมเดล ไม่รับประกันว่าจะเก่งขึ้นเพียงเพิ่ม steps บนข้อมูลตัวอย่างเดิม

ร่างโพสต์หาผู้สนับสนุนอยู่ใน `SUPPORT_DRAFT.md` และยังไม่ได้เผยแพร่
