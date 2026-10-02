// Run: node check_voice.mjs (no microphone, network, or new dependencies).
import assert from 'node:assert/strict';
import { dictate, readAloud } from './app/speech.mjs';

let recognition, utterance, spoken = [], errors = [], ended = 0, cancels = 0;
class Recognition {
  constructor() { recognition = this; }
  start() { this.started = true; }
  abort() { this.aborted = true; }
}
globalThis.webkitSpeechRecognition = Recognition;
const done = () => ended++;
const fail = value => errors.push(value);
const result = text => ({ results: [Object.assign([{ transcript: text }], { isFinal: true })] });
let stop = dictate('th-TH', text => spoken.push(text), fail, done);
assert.equal(recognition.lang, 'th-TH');
assert.equal(recognition.continuous, false);
assert.equal(recognition.started, true);
const late = recognition.onresult;
recognition.onresult(result('สวัสดี'));
assert.deepEqual(spoken, ['สวัสดี']);
assert.equal(ended, 1);
late(result('duplicate'));
assert.equal(spoken.length, 1);
assert.equal(recognition.aborted, true);
stop();

stop = dictate('en-US', text => spoken.push(text), fail, done);
const cancelledResult = recognition.onresult;
stop(); cancelledResult(result('must not enter a new conversation'));
assert.equal(spoken.length, 1);
assert.equal(recognition.onresult, null);
dictate('th-TH', text => spoken.push(text), fail, done);
recognition.onerror({ error: 'not-allowed' });
assert.match(errors.at(-1), /อนุญาต/);
assert.equal(ended, 2);
dictate('th-TH', () => assert.fail('silence produced text'), fail, done);
recognition.onend();
assert.equal(ended, 3);
globalThis.webkitSpeechRecognition = class extends Recognition { start() { throw new Error('start failed'); } };
assert.throws(() => dictate('th-TH', () => {}, fail, done), /start failed/);
assert.equal(recognition.aborted, true);
delete globalThis.webkitSpeechRecognition;
assert.throws(() => dictate('th-TH', () => {}, fail, done), /ไม่รองรับ/);

globalThis.SpeechSynthesisUtterance = class { constructor(text) { this.text = text; } };
globalThis.speechSynthesis = {
  getVoices: () => [{ lang: 'en-US' }, { lang: 'th-TH' }],
  cancel: () => cancels++, speak: value => { utterance = value; },
};
stop = readAloud('ก'.repeat(3500), 'th-TH', fail, done);
assert.equal(utterance.text.length, 3000);
assert.equal(utterance.voice.lang, 'th-TH');
const lateEnd = utterance.onend;
stop(); lateEnd();
assert.equal(ended, 3);
assert.ok(cancels >= 2);
readAloud('hello', 'en-US', fail, done);
utterance.onend();
assert.equal(ended, 4);
readAloud('hello', 'en-US', fail, done);
utterance.onerror();
assert.equal(errors.length, 2);
assert.equal(ended, 5);
delete globalThis.speechSynthesis;
assert.throws(() => readAloud('hello', 'en-US', fail, done), /ไม่รองรับ/);
console.log('PASS: dictation, permission denial, cancellation, late events, startup failure, speech playback and unsupported browsers');
