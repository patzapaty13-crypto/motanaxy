# MOTANAXY Next.js frontend

Run the backend from `C:\AI project`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app_server:app --host 127.0.0.1 --port 8000
```

In another terminal, run the frontend:

```powershell
cd 'C:\AI project\frontend'
npm ci
npm run build
npm run start
```

Open http://127.0.0.1:3000. For editing with automatic reload use `npm run dev` instead of build/start. Do not run dev and start on port 3000 at the same time.

Next.js proxies `/api/*` to `http://127.0.0.1:8000`; model weights remain in the Python backend. To change the backend URL, set `MOTANAXY_BACKEND_URL` before building. The UI renders generated text as escaped React text, preserving indentation; it does not execute generated code or raw HTML. No CDN scripts, external fonts, icon libraries, or model APIs are used by this frontend.

To verify the running frontend/backend together from the project root:

```powershell
.\.venv\Scripts\python.exe check_local_app.py
```

This is a local prototype. Requests are independent, chat history is kept only in the current page, and outputs may be invalid. The 96 documents displayed refer to the latest fine-tuning run, not the parent model's full history. The test score measures syntax only, not functional correctness.

## Voice assistant milestone

Both Local and Cloud screens now provide Thai/English push-to-talk dictation and
on-demand playback of the latest answer. Click **พูดข้อความ**, allow microphone
access, speak, then review the draft and press Send. Voice input never sends a
message or enables Agent mode automatically. **หยุดเสียง**, switching conversations,
submitting a message, leaving the chat panel, or hiding the page cancels active
audio. Drafts keep the existing 4096-character local / 4000-character cloud limit.

The browser's Web Speech APIs provide voice; there is no extra model dependency.
Browser support, installed voices, and microphone permissions vary. Recognition
may use the browser vendor's online service, so voice is not guaranteed offline.
Unsupported browsers keep the regular text interface. Playback reads the first
3000 characters and displays that limit for longer answers.

Run the deterministic voice lifecycle check with `node check_voice.mjs`, then
`npm run build`. For hardware verification, test Thai dictation, deny microphone
permission, stop mid-sentence, and switch conversations while listening. Actual
microphone accuracy and audible output require a browser and device test.

This milestone adds voice interaction to the existing assistant. Cloud mode still
uses its existing conversation history, saved memory, and restricted file tools;
the local model is still an experimental code generator. It does not yet provide
a wake word, computer-wide control, or autonomous self-training.
