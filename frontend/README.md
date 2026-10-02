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
