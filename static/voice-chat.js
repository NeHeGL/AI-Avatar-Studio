/**
 * Hands-free microphone conversation: mic (echo-cancelled) -> simple voice-activity detection -> 16 kHz WAV ->
 * server (/api/chat/turn) -> streamed sentences of reply audio, played one after another.
 */

function encodeWav16k(chunks, inRate) {
  let total = 0;
  for (const c of chunks) total += c.length;
  const merged = new Float32Array(total);
  let o = 0;
  for (const c of chunks) {
    merged.set(c, o);
    o += c.length;
  }
  const ratio = inRate / 16000;
  const n = Math.floor(merged.length / ratio);
  const pcm = new Int16Array(n);
  for (let i = 0; i < n; i++) {
    const a = Math.floor(i * ratio);
    const b = Math.max(a + 1, Math.floor((i + 1) * ratio));
    let s = 0;
    for (let k = a; k < b && k < merged.length; k++) s += merged[k];
    const v = Math.max(-1, Math.min(1, s / (b - a)));
    pcm[i] = v < 0 ? v * 32768 : v * 32767;
  }
  const buf = new ArrayBuffer(44 + pcm.length * 2);
  const dv = new DataView(buf);
  const w = (off, str) => [...str].forEach((ch, i) => dv.setUint8(off + i, ch.charCodeAt(0)));
  w(0, "RIFF");
  dv.setUint32(4, 36 + pcm.length * 2, true);
  w(8, "WAVEfmt ");
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true);
  dv.setUint16(22, 1, true);
  dv.setUint32(24, 16000, true);
  dv.setUint32(28, 32000, true);
  dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true);
  w(36, "data");
  dv.setUint32(40, pcm.length * 2, true);
  new Int16Array(buf, 44).set(pcm);
  return new Blob([buf], { type: "audio/wav" });
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(",")[1]);
    r.onerror = reject;
    r.readAsDataURL(blob);
  });
}

function b64ToBlob(b64, type) {
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return new Blob([arr], { type });
}

/**
 * @param {{name: string, playClip: (blob: Blob, videoUrl?: string) => Promise<void>, onStatus: (s: string) => void,
 *          onLevel: (v: number, threshold: number) => void, onLine: (who: "you"|"avatar", text: string) => void}} o
 */
export function createVoiceChat(o) {
  let ctx = null;
  let stream = null;
  let node = null;
  let superseded = false;
  let busy = false; // avatar is thinking or talking
  // Half-duplex: the avatar's own voice reaches the microphone as loudly as yours (and your avatar sounds like you), so nothing can tell them
  // apart. The microphone is therefore ignored completely from the moment it has something to say until a moment after it finishes. While it is
  // still THINKING, it listens: carry on talking and your parts are joined into one question.
  let turnSeq = 0; // numbers the turns; sentences of a turn that was cut off are never played
  let noticeUntil = 0; // a message ("I did not catch that") stays on screen until then
  let noticePending = false;
  let voiceOutUntil = 0; // while (and just after) the avatar's own voice is playing, the microphone is ignored
  let voiceOut = false;
  let pending = []; // what you said while the avatar was busy, as recorded chunks, sent as one turn when it is done
  let muted = false;
  let stopped = false;
  let resumeAt = 0;
  let floor = 0.004; // running estimate of room noise
  let lastWordsAt = 0; // when the live transcript last grew: words that stop changing mean you have finished
  let liveText = "";
  let sentAt = 0; // when the current question was sent: you can only add to it for a few seconds after
  const MERGE_WINDOW_MS = 3000; // add-ons later than this are a new message, answered after the current reply
  let longestQuietMs = 0; // the longest pause inside the current stretch of sound
  let speaking = false;
  let loud = 0;
  let quietMs = 0;
  let chunks = [];
  let preroll = [];
  let speechMs = 0;
  // Speaking up on its own: if the room stays quiet it says something now and then, less and less often,
  // and after the last attempt it gets bored and stays quiet until the person says something again.
  const quietGaps = o.quietGapsSec || [12, 24, 40, 70, 115, 190, 320]; // seconds of silence before each attempt (+-25%), then it gets bored
  let nudges = 0;
  let lastActive = performance.now();
  let nudgeTimer = 0;
  let lastLoudAt = 0; // any sound above the room noise: it does not start talking on its own while you may be about to speak
  function maybeNudge() {
    if (stopped || busy || muted || speaking || loud > 0 || performance.now() - lastLoudAt < 6000 || nudges >= quietGaps.length) return;
    if (performance.now() - lastActive < quietGaps[nudges] * 1000 * (0.75 + Math.random() * 0.5)) return;
    nudges += 1;
    void runTurn({ nudge: nudges });
  }

  // While it is still THINKING (nothing spoken yet) you can carry on talking: the reply in progress is dropped and your
  // two parts are sent together as one question.
  let turnAudio = null; // the recorded chunks of the turn being answered (null for typed / self-started turns)
  let replyStarted = false;
  let turnNudge = false; // the reply being worked on is one it started by itself (a greeting or speaking up), not an answer to you
  let turnCtl = null;
  let mergePrev = null;
  let heardShown = false; // this turn's "You: ..." line is already on screen
  let replacePrev = false; // the next "You: ..." line replaces the previous one (its turn was dropped and merged)
  async function post(body, onEvent, myTurn) {
    const res = await fetch("/api/chat/turn", {
      signal: turnCtl?.signal,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ canonical_name: o.name, token: o.token, ...body }),
    });
    if (!res.ok || !res.body) throw new Error(`server said ${res.status}`);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    // Play sentences in order while later ones are still being generated.
    let chain = Promise.resolve();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.type === "say") {
          replyStarted = true;
          const blob = b64ToBlob(ev.audio, "audio/wav");
          chain = chain.then(async () => {
            if (stopped || myTurn !== turnSeq) return null; // you cut this reply off: nothing more of it is played
            voiceOut = true;
            try {
              // the words appear when the voice really starts (the page tells us), not when they were written
              return await o.playClip(blob, ev.video_url || "", (secs) => o.onSpeak?.(ev.text, secs));
            } finally {
              voiceOut = false;
              voiceOutUntil = performance.now() + 1500; // its own voice fades out of the room
            }
          });
        } else {
          onEvent?.(ev);
        }
      }
    }
    await chain;
  }

  async function runTurn(body) {
    let engaged = Boolean(body.text); // something typed counts; heard speech counts once it has been accepted (see below)
    if (engaged) nudges = 0;
    busy = true;
    replyStarted = false;
    turnNudge = Boolean(body.nudge);
    heardShown = false;
    turnCtl = new AbortController();
    const myTurn = ++turnSeq;
    let aborted = false;
    o.onStatus("Thinking…");
    let failed = "";
    try {
      await post(body, (ev) => {
        if (ev.type === "superseded") superseded = true;
        if (ev.type === "heard") {
          engaged = true;
          nudges = 0; // you really said something
          heardShown = true;
          o.onLine("you", ev.text, replacePrev); // a merged turn replaces the line of the part that was dropped
          replacePrev = false;
        }
        if (ev.type === "error") failed = ev.message;
        if (ev.type === "ignored") {
          // heard something but not words (or too quiet): say so, instead of silently waiting
          if (ev.reason !== "echo") { // its own voice coming back is dropped quietly
            o.onStatus(ev.reason === "too quiet" ? "I could not hear that. A little louder, please?" : "I did not catch that. Could you say it again?");
            noticeUntil = performance.now() + 3000;
            noticePending = true;
          }
        }
      }, myTurn);
    } catch (e) {
      if (e?.name === "AbortError") aborted = true;
      else failed = e.message || String(e);
    }
    turnAudio = null;
    turnCtl = null;
    if (!aborted) replacePrev = false; // a merged turn that never produced a line (heard as noise) must not make the NEXT line erase an unrelated one
    if (aborted) {
      const cutOff = myTurn !== turnSeq; // you spoke over the reply: your earlier line stays, this is a new thing
      if (heardShown && !cutOff) replacePrev = true;
      busy = false;
      resumeAt = 0;
      return; // you kept talking: the new recording carries both parts (or is the next turn)
    }
    if (superseded) {
      api.stop();
      o.onSuperseded?.();
      return;
    }
    if (failed) o.onStatus(`Problem: ${failed}`);
    resumeAt = performance.now() + 600; // let the room go quiet before listening again
    busy = false;
    if (engaged || body.nudge) lastActive = performance.now(); // room noise that was ignored does not restart the silence clock
    if (pending.length && !failed && !stopped && !muted) {
      const got = pending;
      pending = [];
      sendHeard(got); // you kept talking while it was busy: answer that now
      return;
    }
    pending = [];
    lastState = ""; // report the new state on the next audio tick
  }

  // Live words: while you talk, the speech so far is transcribed locally about once a second and shown as you go.
  let loudMs = 0; // how much of this recording was clearly above the room noise
  let liveWords = 0; // how many real words the live transcript has found in this recording (sound that is words is never thrown away as noise)
  let partialBusy = false;
  let lastPartialAt = 0;
  let partialGen = 0;
  function livePartial() {
    const now = performance.now();
    if (partialBusy || now - lastPartialAt < 800 || speechMs < 500) return;
    lastPartialAt = now;
    partialBusy = true;
    const gen = partialGen;
    const wav = encodeWav16k(chunks.slice(), ctx.sampleRate);
    blobToBase64(wav)
      .then((b64) => fetch("/api/chat/partial", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ audio_base64: b64 }) }))
      .then((r) => r.json())
      .then((j) => {
        if (gen !== partialGen || !speaking || !j.text) return;
        liveWords = Math.max(liveWords, String(j.text).trim().split(/\s+/).length);
        if (j.text !== liveText) { liveText = j.text; lastWordsAt = performance.now(); }
        o.onPartial?.(j.text);
      })
      .catch(() => {})
      .finally(() => {
        partialBusy = false;
      });
  }

  function sendHeard(got) {
    if (mergePrev) {
      const secs = [...mergePrev, ...got].reduce((n, c) => n + c.length, 0) / ctx.sampleRate;
      if (secs < 24) got = [...mergePrev, ...got]; // two parts of one question, as long as the whole fits what the ears can take
      mergePrev = null;
    }
    turnAudio = got;
    sentAt = performance.now();
    const wav = encodeWav16k(got, ctx.sampleRate);
    blobToBase64(wav).then((b64) => runTurn({ audio_base64: b64 }));
  }

  // While the avatar is talking or playing a clip the mic records nothing and typing is blocked; while it thinks, the mic listens.
  let lastState = "";
  function reportState() {
    const ownVoice = voiceOut || performance.now() < voiceOutUntil;
    const wait = busy || ownVoice;
    const st = wait ? "wait" : muted ? "muted" : performance.now() < resumeAt ? "wait" : "listen";
    if (st === lastState) return;
    lastState = st;
    o.onState?.({ listening: st === "listen", canType: !wait && st !== "wait" });
    if (st === "listen" && performance.now() >= noticeUntil) o.onStatus("🎤 Your turn — I'm listening");
  }

  /** Tell the server to stop making the rest of the reply right now (it cannot rely on noticing the closed connection while it is busy). */
  function cancelServerReply() {
    fetch("/api/chat/cancel", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ token: o.token }) }).catch(() => {});
  }

  function startSpeaking() {
    liveWords = 0;
    liveText = "";
    lastWordsAt = performance.now();
    speaking = true;
    loudMs = 0;
    chunks = [...preroll];
    speechMs = 0;
    quietMs = 0;
    longestQuietMs = 0;
    o.onStatus("Listening…");
  }

  function onAudio(e) {
    const x = e.inputBuffer.getChannelData(0);
    let sum = 0;
    for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
    const rms = Math.sqrt(sum / x.length);
    const ms = (x.length / ctx.sampleRate) * 1000;
    const now = performance.now();
    const ownVoice = voiceOut || now < voiceOutUntil;
    const threshold = Math.max(0.01, floor * 3.2);
    const ignoring = muted || ownVoice || (busy && replyStarted) || (!busy && now < resumeAt); // the microphone is not being listened to right now
    o.onLevel(ignoring ? 0 : rms, threshold); // so the level bar stays flat
    reportState();
    if (noticePending && now >= noticeUntil) {
      noticePending = false;
      if (lastState === "listen") o.onStatus("🎤 Your turn — I'm listening");
    }
    if (ignoring) {
      speaking = false;
      partialGen++;
      o.onPartial?.("");
      chunks = [];
      preroll = []; // nothing from while it was muted or talking (its own voice) may leak into the next recording
      loud = 0;
      return;
    }
    if (!speaking) {
      if (rms < threshold) floor = floor * 0.97 + rms * 0.03; // learn the room only while it is quiet
      preroll.push(new Float32Array(x));
      if (preroll.length > 16) preroll.shift(); // ~0.7 s kept from before the trigger so the start of a sentence is not clipped
      loud = rms > threshold ? loud + 1 : 0;
      if (rms > threshold) lastLoudAt = performance.now();
      if (loud >= 2) {
        if (busy && (turnAudio || turnNudge) && !replyStarted && (turnNudge || now - sentAt < MERGE_WINDOW_MS)) {
          if (loud < 4) return; // a click or a bump is not you
          // still thinking, nothing said yet: drop that reply. If it was an answer to you, both parts go together; if it was something it
          // started by itself (a greeting, speaking up), you are simply answered now instead of after all of it
          if (turnAudio) mergePrev = turnAudio;
          cancelServerReply();
          turnCtl?.abort();
        }
        startSpeaking();
      }
      return;
    }
    chunks.push(new Float32Array(x));
    speechMs += ms;
    if (rms > threshold * 1.6) loudMs += ms;
    if (speechMs > 8000 && loudMs < speechMs * 0.2 && liveWords < 3) {
      // "speaking" for 8 s but almost never clearly above the room: that is noise (a fan, a hum), not you. Drop it and learn the room.
      floor = Math.max(floor, rms * 0.5);
      speaking = false;
      partialGen++;
      o.onPartial?.("");
      chunks = [];
      preroll = [];
      loud = 0;
      if (mergePrev) sendHeard([]);
      return;
    }
    // Show what it hears only when the sound is clearly louder than the room (your voice close to the microphone), so background
    // talking does not flash in the message box.
    if (loudMs > speechMs * 0.4) livePartial();
    quietMs = rms < threshold * 0.8 ? quietMs + ms : 0;
    longestQuietMs = Math.max(longestQuietMs, quietMs);
    if (speechMs > 6000 && longestQuietMs < 400 && liveWords < 3) {
      // 6 s of sound with no real pause at all: a TV, a crowd, a fan. A person talking to the avatar pauses. Drop it, and raise the
      // threshold so only a clearly louder voice (yours, close to the microphone) starts a turn.
      floor = Math.max(floor, rms * 0.7);
      speaking = false;
      partialGen++;
      o.onPartial?.("");
      chunks = [];
      preroll = [];
      loud = 0;
      if (mergePrev) sendHeard([]);
      return;
    }
    // The end of what you said: a short remark ends after a short pause (so it is answered while it is still a short sentence), a long
    // one waits a little longer for you to breathe. Words that stopped changing also end it, so a hum or a fan that never goes quiet
    // cannot keep the turn open.
    const spoken = speechMs - quietMs;
    const endQuiet = spoken < 3500 ? 1000 : spoken < 8000 ? 1300 : 1600;
    const wordsDone = liveWords >= 1 && spoken > 1500 && performance.now() - lastWordsAt > 1800;
    if (quietMs >= endQuiet || wordsDone || speechMs > 15000) {
      const got = chunks;
      partialGen++;
      o.onPartial?.("");
      speaking = false;
      chunks = [];
      preroll = [];
      loud = 0;
      if (speechMs - quietMs < 350) {
        if (mergePrev) {
          sendHeard([]); // that was only a click or cough: answer the part that was dropped
          return;
        }
        if (!busy) o.onStatus("Listening… just talk.");
        return; // a click or cough
      }
      if (busy) {
        pending.push(...got); // keep it; answered right after the current reply
        return;
      }
      sendHeard(got);
    }
  }

  const api = {
    async start(deviceId) {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, ...(deviceId ? { deviceId: { exact: deviceId } } : {}) },
      });
      const AC = window.AudioContext || window.webkitAudioContext;
      ctx = new AC();
      await ctx.resume();
      const src = ctx.createMediaStreamSource(stream);
      node = ctx.createScriptProcessor(2048, 1, 1);
      node.onaudioprocess = onAudio;
      const sink = ctx.createGain();
      sink.gain.value = 0;
      src.connect(node);
      node.connect(sink);
      sink.connect(ctx.destination);
      o.onStatus("🎤 Your turn — I'm listening");
      nudgeTimer = setInterval(maybeNudge, 1000);
    },
    /** Open the conversation: the model writes a proper hello for this person and this moment. */
    greet: () => runTurn({ nudge: 9 }),
    /** Restart the silence clock (after the greeting). */
    touch() {
      lastActive = performance.now();
      nudges = 0;
    },
    sendText: (text) => {
      if (busy || voiceOut || performance.now() < voiceOutUntil) return; // wait for your turn
      return runTurn({ text });
    },
    setMuted(m) {
      muted = m;
      o.onStatus(m ? "Mic muted." : "Listening… just talk.");
    },
    get muted() {
      return muted;
    },
    stop() {
      stopped = true;
      clearInterval(nudgeTimer);
      try {
        node?.disconnect();
        stream?.getTracks().forEach((t) => t.stop());
        ctx?.close();
      } catch {
        /* ignore */
      }
    },
  };
  return api;
}
