'use client';

import { useEffect, useRef, useState } from 'react';
import { dictate, readAloud } from './speech.mjs';

export default function VoiceControls({ setPrompt, answer, disabled, maxLength }) {
  const [supported, setSupported] = useState({ input: false, output: false });
  const [lang, setLang] = useState('th-TH');
  const [mode, setMode] = useState('idle');
  const [error, setError] = useState('');
  const cancel = useRef(null);

  function stop() {
    cancel.current?.(); cancel.current = null; setMode('idle');
  }
  useEffect(() => {
    setSupported({ input: !!(window.SpeechRecognition || window.webkitSpeechRecognition),
      output: !!(window.speechSynthesis && window.SpeechSynthesisUtterance) });
    const leave = () => stop();
    const hide = () => { if (document.hidden) stop(); };
    window.addEventListener('pagehide', leave);
    document.addEventListener('visibilitychange', hide);
    return () => {
      cancel.current?.();
      window.removeEventListener('pagehide', leave);
      document.removeEventListener('visibilitychange', hide);
    };
  }, []);
  useEffect(() => { if (disabled) stop(); }, [disabled]);

  function start(next) {
    stop(); setError(''); setMode(next);
    const done = () => { cancel.current = null; setMode('idle'); };
    try {
      cancel.current = next === 'listening'
        ? dictate(lang, text => {
          setPrompt(previous => `${previous}${previous ? ' ' : ''}${text}`.slice(0, maxLength));
        }, setError, done)
        : readAloud(answer, lang, setError, done);
    } catch (cause) { setError(cause.message); done(); }
  }

  return <section className="voice-controls" aria-label="สนทนาด้วยเสียง">
    <div className="voice-actions">
      <label>ภาษาเสียง <select value={lang} disabled={mode !== 'idle'} onChange={e => setLang(e.target.value)}>
        <option value="th-TH">ไทย</option><option value="en-US">English</option>
      </select></label>
      <button type="button" className="plain-button" disabled={disabled || !supported.input || mode !== 'idle'} onClick={() => start('listening')}>พูดข้อความ</button>
      <button type="button" className="plain-button" disabled={disabled || !supported.output || !answer || mode !== 'idle'} onClick={() => start('speaking')}>ฟังคำตอบล่าสุด</button>
      {mode !== 'idle' && <button type="button" className="plain-button" onClick={stop}>หยุดเสียง</button>}
      <span role="status">{mode === 'listening' ? 'กำลังฟัง…' : mode === 'speaking' ? 'กำลังอ่าน…' : 'ตรวจข้อความก่อนกดส่ง'}</span>
    </div>
    <p className="muted">{supported.input ? 'เปิดไมค์เมื่อกดพูดเท่านั้น · เบราว์เซอร์อาจส่งเสียงไปบริการถอดเสียงออนไลน์' : 'เบราว์เซอร์นี้ไม่รองรับการรับเสียง ใช้การพิมพ์ได้ตามปกติ'}{answer?.length > 3000 ? ' · อ่านเฉพาะ 3,000 ตัวอักษรแรก' : ''}</p>
    {error && <p className="error-message" role="alert">{error}</p>}
  </section>;
}
