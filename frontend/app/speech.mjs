// One-shot dictation fills a draft; it never sends a chat request.
export function dictate(lang, onText, onError, onEnd) {
  const Recognition = globalThis.SpeechRecognition || globalThis.webkitSpeechRecognition;
  if (!Recognition) throw new Error('เบราว์เซอร์นี้ไม่รองรับการรับเสียง กรุณาพิมพ์ข้อความ');
  const recognition = new Recognition();
  recognition.lang = lang;
  recognition.continuous = false;
  recognition.interimResults = false;
  let active = true;
  function cancel() {
    if (!active) return;
    active = false;
    recognition.onresult = recognition.onerror = recognition.onend = null;
    recognition.abort();
  }
  recognition.onresult = event => {
    if (!active) return;
    const text = Array.from(event.results).filter(result => result.isFinal)
      .map(result => result[0].transcript).join(' ').trim();
    if (text) { cancel(); onText(text); onEnd(); }
  };
  recognition.onerror = event => {
    if (!active) return;
    cancel();
    const messages = {
      'not-allowed': 'ไม่ได้รับอนุญาตใช้ไมโครโฟน เปิดสิทธิ์ในเบราว์เซอร์แล้วลองใหม่',
      'service-not-allowed': 'เบราว์เซอร์ไม่อนุญาตบริการรับเสียง กรุณาพิมพ์ข้อความ',
      'audio-capture': 'ไม่พบไมโครโฟนที่ใช้งานได้',
      'no-speech': 'ไม่ได้ยินเสียงพูด ลองใหม่ได้ครับ',
      network: 'บริการรับเสียงเชื่อมต่อไม่ได้ ตรวจอินเทอร์เน็ตแล้วลองใหม่',
    };
    onError(messages[event.error] || 'รับเสียงไม่สำเร็จ กรุณาลองใหม่หรือพิมพ์ข้อความ');
    onEnd();
  };
  recognition.onend = () => { if (active) { cancel(); onEnd(); } };
  try { recognition.start(); } catch (error) { cancel(); throw error; }
  return cancel;
}

export function readAloud(text, lang, onError, onEnd) {
  const synth = globalThis.speechSynthesis;
  if (!synth || !globalThis.SpeechSynthesisUtterance) throw new Error('เบราว์เซอร์นี้ไม่รองรับการอ่านออกเสียง');
  // ponytail: read the first 3000 characters; chunk playback if longer narration is needed.
  const utterance = new SpeechSynthesisUtterance(text.slice(0, 3000));
  utterance.lang = lang;
  const voice = synth.getVoices().find(item => item.lang.toLowerCase().startsWith(lang.slice(0, 2)));
  if (voice) utterance.voice = voice;
  let active = true;
  function cancel() {
    if (!active) return;
    active = false;
    utterance.onend = utterance.onerror = null;
    synth.cancel();
  }
  utterance.onend = () => { if (active) { cancel(); onEnd(); } };
  utterance.onerror = () => {
    if (!active) return;
    cancel(); onError('อ่านออกเสียงไม่ได้ ตรวจว่าเครื่องมีเสียงสำหรับภาษาที่เลือก'); onEnd();
  };
  try { synth.cancel(); synth.speak(utterance); } catch (error) { cancel(); throw error; }
  return cancel;
}
