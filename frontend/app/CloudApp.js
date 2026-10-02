'use client';

import { useEffect, useRef, useState } from 'react';
import VoiceControls from './VoiceControls';

async function api(path, method = 'GET', body) {
  const response = await fetch(`/api/${path}`, { method, credentials: 'same-origin',
    headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const value = await response.json();
  if (!response.ok) { const error = new Error(value.error || 'Request failed'); error.status = response.status; throw error; }
  return value;
}

function RichText({ value }) {
  // Render model output as text, never HTML. Only fenced code gets a separate block.
  return value.split(/(```[\s\S]*?```)/g).map((part, index) => part.startsWith('```')
    ? <Code key={index} content={part.replace(/^```[^\n]*\n?/, '').replace(/```$/, '')} />
    : <div key={index} className="answer-prose">{part}</div>);
}

function Code({ content }) {
  const [copied, setCopied] = useState(false);
  return <div className="code-output"><div className="code-toolbar"><span>CODE · ตรวจสอบก่อนนำไปใช้</span><button onClick={async () => {
    try { await navigator.clipboard.writeText(content); setCopied(true); } catch { setCopied(false); }
  }}>{copied ? 'Copied' : 'Copy'}</button></div><pre><code>{content}</code></pre></div>;
}

export default function CloudApp() {
  const [model, setModel] = useState(null);
  const [locked, setLocked] = useState(false);
  const [key, setKey] = useState('');
  const [panel, setPanel] = useState('chat');
  const [sessions, setSessions] = useState([]);
  const [session, setSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [settings, setSettings] = useState({ memory: '', collect: false });
  const [files, setFiles] = useState([]);
  const [revisions, setRevisions] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [samples, setSamples] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [agent, setAgent] = useState(false);
  const [temperature, setTemperature] = useState(0.5);
  const [tokens, setTokens] = useState(768);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [voiceEpoch, setVoiceEpoch] = useState(0);
  const bottom = useRef(null);
  const input = useRef(null);
  const inFlight = useRef(false);

  async function refresh() {
    const [info, preferences, history, documents, training] = await Promise.all([
      api('model'), api('settings'), api('sessions'), api('files'), api('samples'),
    ]);
    setModel(info); setSettings({ ...preferences, collect: !!preferences.collect });
    setSessions(history); setFiles(documents); setSamples(training); setLocked(false);
  }
  function fail(cause) {
    setError(cause.message);
    if (cause.status === 401) { setLocked(true); setModel(null); setMessages([]); setSession(null); setSettings({memory:'',collect:false}); setFiles([]); setSamples([]); }
  }
  useEffect(() => { refresh().catch(fail); }, []);
  useEffect(() => { if (messages.length) bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [messages]);

  async function login(event) {
    event.preventDefault(); setBusy(true); setError('');
    try { await api('login', 'POST', { key }); setKey(''); await refresh(); }
    catch (cause) { fail(cause); }
    finally { setBusy(false); }
  }

  async function openSession(value) {
    if (busy) return;
    setVoiceEpoch(previous => previous + 1);
    try {
      const history = await api(`sessions/${value}`);
      setSession(value); setMessages(history.map(row => ({ ...row, text: row.content, trace: JSON.parse(row.trace || '[]') })));
      setPanel('chat'); setError(''); setNotice('');
    } catch (cause) { fail(cause); }
  }

  async function send(event) {
    event.preventDefault();
    if (inFlight.current || !prompt.trim()) return;
    const question = prompt.trim();
    inFlight.current = true; setBusy(true); setError(''); setNotice(''); setPrompt('');
    setMessages(previous => [...previous, { role: 'user', text: question }]);
    try {
      const response = await api('chat', 'POST', { message: question, session_id: session,
        agent, temperature, max_tokens: tokens });
      setSession(response.session_id);
      setMessages(previous => [...previous, { id: response.message_id, role: 'assistant', text: response.response,
        trace: response.trace, elapsed: response.elapsed_seconds }]);
      const [info, history, docs] = await Promise.all([api('model'), api('sessions'), api('files')]);
      setModel(info); setSessions(history); setFiles(docs);
    } catch (cause) { fail(cause); setPrompt(question); }
    finally { setBusy(false); inFlight.current = false; input.current?.focus(); }
  }

  async function saveSettings() {
    try { await api('settings', 'PUT', settings); setNotice('บันทึกความจำแล้ว ใช้ในการตอบครั้งถัดไป ไม่ได้ฝึกน้ำหนักโมเดล'); setError(''); }
    catch (cause) { fail(cause); }
  }
  async function approve(messageId) {
    try { await api('samples', 'POST', { message_id: messageId }); setSamples(await api('samples')); setNotice('เพิ่มในคิวที่คุณอนุมัติแล้ว ยังไม่ผ่านการทดสอบและยังไม่ได้เทรน'); }
    catch (cause) { fail(cause); }
  }

  if (locked) return <main className="login-page"><form className="login-card" onSubmit={login}>
    <div className="brand"><span className="brand-icon">◈</span><strong>MOTANAXY<span className="brand-period">.</span></strong></div>
    <h1>พื้นที่ AI ส่วนตัว</h1><p>Qwen ประมวลผลบน Cloudflare · เครื่องคุณใช้เพียงเบราว์เซอร์</p>
    <label htmlFor="access-key">รหัสเข้าใช้งาน MOTANAXY</label><input id="access-key" type="password" autoComplete="current-password" required value={key} onChange={e => setKey(e.target.value)} />
    {error && <p className="error-message" role="alert">{error}</p>}<button className="primary-button" disabled={busy}>{busy ? 'กำลังเข้าสู่ระบบ…' : 'เข้าสู่พื้นที่ของฉัน →'}</button>
    <p className="muted">แชตและความจำถูกเก็บในบัญชี Cloudflare ของเจ้าของระบบ รหัสนี้ไม่ใช่รหัสผ่าน Cloudflare</p>
  </form></main>;

  return <div className="app-shell cloud-shell">
    <aside className="sidebar"><a className="brand" href="/"><span className="brand-icon">◈</span><span><strong>MOTANAXY<span className="brand-period">.</span></strong><small>PERSONAL CLOUD AGENT</small></span></a>
      <button className="new-session" disabled={busy} onClick={() => { setVoiceEpoch(previous => previous + 1); setSession(null); setMessages([]); setPanel('chat'); setPrompt(''); setError(''); setNotice(''); }}>＋ New conversation</button>
      <nav className="cloud-nav" aria-label="เมนูหลัก">{[['chat','แชต'],['memory','ความจำ'],['files','ไฟล์ทดลอง'],['learning','คิวเรียนรู้']].map(([name, label]) => <button key={name} aria-current={panel === name ? 'page' : undefined} onClick={() => setPanel(name)}>{label}{name === 'files' && <span>{files.length}</span>}{name === 'learning' && <span>{samples.length}</span>}</button>)}</nav>
      <section className="model-card"><div className="section-label">CLOUD MODEL <span className={`status-dot ${model ? 'online' : ''}`} /></div><h2>{model?.run ?? 'Connecting…'}</h2><p>{model?.provider ?? 'Cloudflare Workers AI'}</p><dl className="model-stats"><div><dt>ประมวลผล</dt><dd>Cloud GPU</dd></div><div><dt>Model calls วันนี้</dt><dd>{model?.calls_today ?? '—'} / {model?.call_limit ?? 40}</dd></div><div><dt>Local model</dt><dd>Not loaded</dd></div></dl><span className="experiment-tag">PRIVATE EXPERIMENT</span></section>
      <div className="history-list"><div className="section-label">CONVERSATIONS</div>{sessions.slice(0, 15).map(item => <div className="history-item" key={item.id}><button disabled={busy} onClick={() => openSession(item.id)} title={item.title}>{item.title}</button><button disabled={busy} aria-label={`ลบบทสนทนา ${item.title}`} onClick={async () => {
        if (!window.confirm('ลบบทสนทนานี้และตัวอย่างฝึกที่คัดจากบทสนทนานี้?')) return;
        try { await api(`sessions/${item.id}`, 'DELETE'); if (session === item.id) { setMessages([]); setSession(null); } await refresh(); } catch (cause) { fail(cause); }
      }}>×</button></div>)}</div>
      <footer className="sidebar-footer"><div>Cloudflare · Qwen<small>Memory ≠ training weights</small></div><button className="plain-button" disabled={busy} onClick={async () => { try { await api('logout', 'POST', {}); fail({status:401,message:''}); } catch (cause) { fail(cause); } }}>ออก</button></footer>
    </aside>

    <main className="workspace"><header className="topbar"><div className="breadcrumb">MOTANAXY <span>/</span><strong>{panel === 'chat' ? agent ? 'Agent workspace' : 'Conversation' : panel}</strong></div><div className="connection online" role="status"><span className="status-dot" />{model ? 'Cloud connected' : 'Connecting…'}</div></header>
      {error && <p className="error-message cloud-alert" role="alert">{error}</p>}{notice && <p className="notice cloud-alert" role="status">{notice}</p>}

      {panel === 'chat' && <><div className="conversation">
        {!messages.length && <section className="welcome"><div className="welcome-symbol">◈</div><div className="eyebrow">YOUR ASSISTANT. ON CLOUD.</div><h1>คุย วางแผน ลงมือทำ<span>แล้วเรียนรู้จากผลจริง</span></h1><p className="welcome-description">Qwen ช่วยคิดและเขียนโค้ด · ความจำปรับคำตอบให้เข้ากับคุณ<br />เปิด Agent เพื่อให้ทำงานกับไฟล์ในพื้นที่ทดลองบนคลาวด์</p>
          <div className="suggestions">{[
            ['ออกแบบระบบ','ช่วยออกแบบระบบนับสินค้าเข้าออกสำหรับร้านเล็ก ใช้ Next.js เริ่มจากถามสิ่งที่จำเป็น 3 ข้อ',false],
            ['ให้ AI สร้างไฟล์','สร้างไฟล์ sum.js ที่ export ฟังก์ชัน add(a,b) แล้วตรวจไวยากรณ์ด้วยเครื่องมือ',true],
            ['ช่วยแก้โค้ด','อธิบายว่าทำไม Python ใช้ mutable default argument แล้วเกิดบั๊ก พร้อมวิธีแก้สั้น ๆ',false],
          ].map(([title,value,mode]) => <button className="suggestion" key={title} onClick={() => { setPrompt(value); setAgent(mode); input.current?.focus(); }}><strong>{title}</strong><small>{mode ? 'Agent · Cloud files' : 'Chat · Thai + English'}</small></button>)}</div><p className="honesty-note">AI อาจผิดพลาด · เครื่องมือตรวจได้เฉพาะไวยากรณ์ JSON/JS ไม่ได้รันโปรแกรม</p></section>}
        <div className="message-list">{messages.map((message, index) => <article className={`message ${message.role}`} key={message.id || index}><div className="avatar">{message.role === 'user' ? 'Y' : '◈'}</div><div className="message-content"><div className="message-label">{message.role === 'user' ? 'YOU' : 'MOTANAXY · QWEN'}{message.elapsed != null && <span>{message.elapsed.toFixed(1)}s · cloud inference</span>}</div><RichText value={message.text} />{message.trace?.length > 0 && <details className="tool-trace"><summary>การใช้เครื่องมือ {message.trace.length} ครั้ง</summary>{message.trace.map((step, n) => <div key={n}><strong>{step.tool}</strong><pre>{JSON.stringify(step.result, null, 2)}</pre></div>)}</details>}{message.role === 'assistant' && settings.collect && message.id && <button className="plain-button" onClick={() => approve(message.id)}>คัดคำตอบนี้เข้าคิวข้อมูลฝึก</button>}</div></article>)}</div>
        {busy && <div className="generating" role="status"><span className="pulse" />กำลังประมวลผลบนคลาวด์ {agent ? '· สูงสุด 4 รอบต่อคำขอ' : ''}</div>}<div ref={bottom} />
      </div><div className="composer-area"><div className="cloud-settings"><label><input type="checkbox" checked={agent} disabled={busy} onChange={e => setAgent(e.target.checked)} /> Agent: อนุญาตแก้ไฟล์ทดลอง</label><label>คำตอบ<select aria-label="ความยาวคำตอบ" value={tokens} onChange={e => setTokens(Number(e.target.value))}><option value={384}>สั้น</option><option value={768}>ปกติ</option><option value={1536}>ยาว</option></select></label><label>Temp <input aria-label="Temperature" type="number" step="0.1" min="0.1" max="1.2" value={temperature} onChange={e => setTemperature(Number(e.target.value))} /></label></div>
        <VoiceControls key={voiceEpoch} setPrompt={setPrompt} answer={messages.filter(message => message.role === 'assistant').at(-1)?.text || ''} disabled={busy} maxLength={4000} /><form className="composer" onSubmit={send}><label htmlFor="cloud-prompt" className="sr-only">พิมพ์คำถาม</label><textarea ref={input} id="cloud-prompt" maxLength={4000} rows={2} value={prompt} placeholder="บอกสิ่งที่อยากทำ หรือถามต่อจากบทสนทนาเดิม…" onChange={e => setPrompt(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) send(e); }} /><div className="composer-bottom"><span>{agent ? 'Agent · Private cloud files' : 'Chat · Conversation memory'}</span><button type="submit" className="send-button" disabled={busy || !model || !prompt.trim()} aria-label="ส่งข้อความ">↑</button></div></form><p className="composer-note">ประวัติอยู่บน Cloudflare · ไม่ใช้แชตฝึกน้ำหนักอัตโนมัติ · ถึงโควตาจะหยุด</p></div></>}

      {panel === 'memory' && <section className="cloud-panel"><h1>ความจำของคุณ</h1><p>บอกภาษา สไตล์ หรือข้อกำหนดโปรเจกต์ ข้อมูลนี้ถูกส่งให้โมเดลทุกครั้ง แก้ไขหรือลบได้ ไม่ใช่การเปลี่ยนน้ำหนักโมเดล</p><label htmlFor="memory">สิ่งที่อยากให้จำ</label><textarea id="memory" rows={10} maxLength={3000} value={settings.memory} placeholder="ตอบภาษาไทยแบบกระชับ ใช้ Next.js และเน้นอธิบายสำหรับมือใหม่…" onChange={e => setSettings({...settings,memory:e.target.value})} /><div className="panel-actions"><button className="primary-button" onClick={saveSettings}>บันทึกความจำ</button><button className="plain-button" onClick={async () => { try { await api('settings','PUT',{...settings,memory:''}); setSettings({...settings,memory:''}); setNotice('ลบความจำแล้ว'); } catch(cause) {fail(cause);} }}>ลบความจำทั้งหมด</button></div><p className="muted">พื้นที่นี้สำหรับเจ้าของระบบหนึ่งคน ไม่ใช่ระบบหลายบัญชี อย่าเก็บรหัสผ่านหรือ API key</p></section>}

      {panel === 'files' && <section className="cloud-panel"><h1>ไฟล์ทดลองบนคลาวด์</h1><p>Agent เข้าถึงได้เฉพาะไฟล์ชุดนี้ ไม่ได้อ่านหรือเปลี่ยนไฟล์ในคอมพิวเตอร์ของคุณ เก็บย้อนหลังสูงสุด 10 รุ่นต่อไฟล์</p>{files.length === 0 && <p>ยังไม่มีไฟล์ เปิด Agent แล้วสั่งสร้างไฟล์แรกได้เลย</p>}{files.map(file => <details className="file-card" key={file.path}><summary>{file.path} <small>{file.content.length.toLocaleString()} chars</small></summary><Code content={file.content} /><button className="plain-button" onClick={async () => { try {setSelectedFile(file.path); setRevisions(await api(`revisions?path=${encodeURIComponent(file.path)}`));}catch(cause){fail(cause);} }}>ดูเวอร์ชันก่อนหน้า</button>{selectedFile === file.path && <div>{revisions.length === 0 ? <p>ยังไม่มีเวอร์ชันก่อนหน้า</p> : revisions.map(revision => <details key={revision.id}><summary>{revision.created}</summary><Code content={revision.content} /><button className="plain-button" disabled={busy} onClick={async () => {if(!window.confirm('คืนไฟล์เป็นเวอร์ชันนี้? รุ่นปัจจุบันจะเก็บเป็นประวัติ'))return; try {await api('restore','POST',{id:revision.id});setFiles(await api('files'));setNotice('คืนไฟล์แล้ว');}catch(cause){fail(cause);}}}>คืนเวอร์ชันนี้</button></details>)}</div>}</details>)}</section>}

      {panel === 'learning' && <section className="cloud-panel"><h1>คิวข้อมูลเรียนรู้</h1><p>เลือกเฉพาะคำตอบที่คุณอนุญาตและตรวจแล้ว การเพิ่มในคิวไม่ได้แปลว่าคำตอบถูกต้องหรือโมเดลถูกเทรนแล้ว</p><label className="opt-in"><input type="checkbox" checked={settings.collect} onChange={async e => {const next={...settings,collect:e.target.checked};try{await api('settings','PUT',next);setSettings(next);}catch(cause){fail(cause);}}} />อนุญาตให้ฉันคัดคำตอบจากแชตเข้าคิว</label><p className="muted">มีตัวกรอง secret แบบพื้นฐาน แต่ไม่รับประกันว่าจะพบข้อมูลลับทุกชนิด ตรวจทานเองก่อนคัดหรือส่งออก</p><div className="panel-actions"><button className="primary-button" disabled={!samples.length} onClick={() => {
        const blob=new Blob([samples.map(row => JSON.stringify({messages:[{role:'user',content:row.prompt},{role:'assistant',content:row.answer}],source:row.source,status:row.status})).join('\n')],{type:'application/x-ndjson'});
        const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download='motanaxy-approved-examples.jsonl'; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
      }}>Export JSONL ({samples.length})</button><button className="plain-button" disabled={!samples.length} onClick={async()=>{if(!window.confirm('ลบข้อมูลที่คัดเข้าคิวทั้งหมด? แชตต้นฉบับยังอยู่'))return;try{await api('samples','DELETE');setSamples([]);}catch(cause){fail(cause);}}}>ล้างคิว</button></div>{samples.map(row=><details className="file-card" key={row.id}><summary>{row.prompt.slice(0,100)}</summary><RichText value={row.answer}/><p className="muted">{row.status}</p></details>)}</section>}
    </main>
  </div>;
}
