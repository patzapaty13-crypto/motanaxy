'use client';

import { useEffect, useRef, useState } from 'react';
import CloudApp from './CloudApp';

function Icon({ name, size = 20 }) {
  const paths = {
    chip: <><rect x="6" y="6" width="12" height="12" rx="3" /><rect x="9" y="9" width="6" height="6" rx="1" /><path d="M9 3v3m6-3v3M9 18v3m6-3v3M3 9h3m-3 6h3m12-6h3m-3 6h3" /></>,
    plus: <path d="M12 5v14M5 12h14" />,
    arrow: <path d="m5 12 7-7 7 7M12 5v14" />,
    code: <><path d="m7 7-5 5 5 5m10-10 5 5-5 5M14 4l-4 16" /></>,
    copy: <><rect x="8" y="8" width="12" height="12" rx="2" /><path d="M15 8V4H4v11h4" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    refresh: <><path d="M20 7v5h-5M4 17v-5h5" /><path d="M5.5 7a7 7 0 0 1 12-2L20 8M4 16l2.5 3a7 7 0 0 0 12-2" /></>,
    terminal: <><path d="m5 6 6 6-6 6M13 18h6" /></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

const suggestions = [
  { title: 'เริ่มจากฟังก์ชันง่าย ๆ', label: 'Python · Code completion', prompt: 'def add(a, b):' },
  { title: 'จัดการข้อมูลในลิสต์', label: 'Python · Data structures', prompt: '# ลบค่าซ้ำโดยคงลำดับ' },
  { title: 'ทดลองโจทย์ภาษาอังกฤษ', label: 'Python · Algorithms', prompt: '# Find the greatest common divisor of integers.' },
];

function CodeOutput({ code }) {
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setCopyError(false);
    } catch {
      setCopyError(true);
    }
  }
  return <div className="code-output">
    <div className="code-toolbar"><span><span className="code-dot" /> GENERATED TEXT</span><button type="button" onClick={copy} aria-label="คัดลอกโค้ด"><Icon name={copied ? 'check' : 'copy'} size={14} />{copyError ? 'เลือกข้อความเพื่อคัดลอก' : copied ? 'Copied' : 'Copy'}</button></div>
    <pre><code>{code}</code></pre>
  </div>;
}

function LocalHome() {
  const [model, setModel] = useState(null);
  const [connection, setConnection] = useState('connecting');
  const [messages, setMessages] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [temperature, setTemperature] = useState(0.6);
  const [tokens, setTokens] = useState(250);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const input = useRef(null);
  const bottom = useRef(null);
  const inFlight = useRef(false);

  async function refreshModel() {
    setConnection('connecting');
    try {
      const response = await fetch('/api/model', { signal: AbortSignal.timeout(10000), cache: 'no-store' });
      if (!response.ok) throw new Error('Backend unavailable');
      setModel(await response.json());
      setConnection('online');
    } catch {
      setConnection('offline');
    }
  }

  useEffect(() => { refreshModel(); }, []);
  useEffect(() => {
    if (messages.length || busy) bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, busy]);

  async function submit(event) {
    event?.preventDefault();
    if (!prompt.trim() || inFlight.current) return;
    const text = prompt.trim();
    inFlight.current = true;
    setBusy(true);
    setError('');
    setPrompt('');
    setMessages(previous => [...previous, { role: 'user', text }]);
    try {
      const response = await fetch('/api/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, temperature, max_tokens: tokens, top_k: 40 }),
        signal: AbortSignal.timeout(60000),
      });
      if (!response.ok) throw new Error(`Request failed (${response.status})`);
      const data = await response.json();
      setMessages(previous => [...previous, { role: 'assistant', text: data.code ?? data.raw_code, elapsed: data.elapsed_seconds }]);
      setConnection('online');
    } catch (cause) {
      setError(cause.name === 'TimeoutError' ? 'โมเดลใช้เวลานานเกินไป ลองลด Max bytes แล้วส่งใหม่' : 'เชื่อมต่อโมเดลไม่ได้ ตรวจว่า backend รันอยู่ที่พอร์ต 8000 แล้วลองอีกครั้ง');
      setPrompt(text);
      setConnection('offline');
    } finally {
      inFlight.current = false;
      setBusy(false);
      input.current?.focus();
    }
  }

  const online = connection === 'online';
  const score = model?.evaluation;

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="/" aria-label="MOTANAXY home"><span className="brand-icon"><Icon name="chip" size={27} /></span><span><strong>MOTANAXY<span className="brand-period">.</span></strong><small>LOCAL CODE LAB</small></span></a>
      <button className="new-session" type="button" disabled={busy} onClick={() => { setMessages([]); setPrompt(''); setError(''); input.current?.focus(); }}><Icon name="plus" size={18} />New session<span>↗</span></button>

      <section className="model-card" aria-label="Active model">
        <div className="section-label">ACTIVE MODEL <span className={`status-dot ${online ? 'online' : ''}`} /></div>
        <h2>{model?.run ?? 'Connecting…'}</h2><p>Byte-level transformer</p>
        <dl className="model-stats"><div><dt>Parameters</dt><dd>{model?.parameters?.toLocaleString() ?? '—'}</dd></div><div><dt>Context</dt><dd>{model?.context_bytes ?? '—'} bytes</dd></div><div><dt>Training documents</dt><dd>{model?.training_documents ?? '—'}</dd></div><div><dt>Compute</dt><dd>Local CPU</dd></div></dl>
        <span className="experiment-tag"><span />EXPERIMENTAL</span>
      </section>

      <section className="controls" aria-label="Generation settings"><div className="section-label">GENERATION SETTINGS</div>
        <div className="slider-control"><label htmlFor="temperature">Temperature <output>{temperature.toFixed(1)}</output></label><input id="temperature" type="range" min="0.1" max="1.5" step="0.1" value={temperature} onChange={e => setTemperature(Number(e.target.value))} /><div className="range-labels"><span>Focused</span><span>Exploratory</span></div></div>
        <div className="slider-control"><label htmlFor="tokens">Max bytes <output>{tokens}</output></label><input id="tokens" type="range" min="50" max="500" step="50" value={tokens} onChange={e => setTokens(Number(e.target.value))} /><div className="range-labels"><span>50</span><span>500</span></div></div>
      </section>

      {score && <section className="training-card"><div className="section-label">LATEST EXPERIMENT</div><div className="score-row"><span>Held-out loss</span><strong>{score.before_loss.toFixed(2)} <span>→</span> {score.after_loss.toFixed(2)}</strong></div><p>Syntax check: {score.syntax_passes}/{score.prompts} prompts<br />ไม่ใช่คะแนนความถูกต้องของโค้ด</p></section>}
      <footer className="sidebar-footer"><span className="local-symbol">◈</span><div>Runs on your machine<small>No model API calls</small></div><span className="version">v0.1</span></footer>
    </aside>

    <main className="workspace">
      <header className="topbar"><div className="breadcrumb"><Icon name="terminal" size={18} /><span>Playground</span><span className="breadcrumb-slash">/</span><strong>New session</strong></div><div className={`connection ${connection}`} role="status"><span className="status-dot" />{online ? 'Model connected' : connection === 'connecting' ? 'Connecting' : 'Backend offline'}<button type="button" onClick={refreshModel} aria-label="ตรวจการเชื่อมต่อใหม่"><Icon name="refresh" size={14} /></button></div></header>

      <div className="conversation">
        {messages.length === 0 ? <section className="welcome">
          <div className="welcome-symbol"><Icon name="code" size={34} /></div>
          <div className="eyebrow">YOUR MODEL. YOUR MACHINE.</div>
          <h1>ทุกไอเดีย เริ่มที่<span>โค้ดหนึ่งบรรทัด</span></h1>
          <p className="welcome-description">พื้นที่ทดลองโมเดลที่เราฝึกเอง<br />เริ่มด้วยฟังก์ชัน Python สั้น ๆ แล้วสำรวจสิ่งที่โมเดลเรียนรู้</p>
          <div className="suggestions">{suggestions.map(item => <button type="button" className="suggestion" key={item.title} onClick={() => { setPrompt(item.prompt); input.current?.focus(); }}><span className="suggestion-icon"><Icon name="code" size={17} /></span><strong>{item.title}</strong><small>{item.label}</small><span className="suggestion-arrow">↗</span></button>)}</div>
          <div className="honesty-note"><span>i</span>โมเดลยังอยู่ในระยะทดลอง ผลลัพธ์อาจอ่านไม่รู้เรื่องหรือรันไม่ได้</div>
        </section> : <div className="message-list">{messages.map((message, index) => <article className={`message ${message.role}`} key={index}><div className="avatar">{message.role === 'user' ? 'Y' : <Icon name="chip" size={18} />}</div><div className="message-content"><div className="message-label">{message.role === 'user' ? 'YOU' : 'MOTANAXY'}{message.elapsed != null && <span>{message.elapsed.toFixed(1)}s · local inference</span>}</div>{message.role === 'user' ? <p className="user-text">{message.text}</p> : <><CodeOutput code={message.text} /><p className="output-note">ข้อความที่โมเดลสร้างจริง · ยังไม่ได้ตรวจความถูกต้อง</p></>}</div></article>)}</div>}
        {busy && <div className="generating" role="status"><span className="pulse" />โมเดลกำลังสร้างข้อความบนเครื่องนี้…</div>}
        <div ref={bottom} />
      </div>

      <div className="composer-area">{error && <p className="error-message" role="alert">{error}</p>}
        <form className="composer" onSubmit={submit}><label htmlFor="prompt" className="sr-only">คำถามหรือโค้ดที่ต้องการให้โมเดลต่อ</label><textarea ref={input} id="prompt" value={prompt} maxLength={4096} rows={2} placeholder="พิมพ์โจทย์ หรือเริ่มด้วย def add(a, b):" onChange={e => setPrompt(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); submit(); } }} /><div className="composer-bottom"><span><Icon name="code" size={14} />Python <span className="composer-separator">·</span> {online ? 'Local model' : 'Waiting for backend'}</span><button className="send-button" type="submit" disabled={busy || !prompt.trim() || !online} aria-label="ส่งให้โมเดลสร้างโค้ด"><Icon name="arrow" size={20} /></button></div></form>
        <p className="composer-note"><span>Enter เพื่อส่ง · Shift + Enter ขึ้นบรรทัดใหม่</span><span>แต่ละคำขอประมวลผลแยกกัน</span></p>
      </div>
    </main>
  </div>;
}

export default function Home() {
  return process.env.NEXT_PUBLIC_MOTANAXY_CLOUD === '1' ? <CloudApp /> : <LocalHome />;
}
