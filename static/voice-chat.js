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
  let busy = false; // avatar is thinking or talking: keep listening, but save what you say for the next turn
  let voiceOutUntil = 0; // while (and just after) the avatar's own voice is playing, only clearly louder speech counts
  let voiceOut = false;
  let pending = []; // what you said while the avatar was busy, as recorded chunks, sent as one turn when it is done
  let muted = false;
  let stopped = false;
  let resumeAt = 0;
  let floor = 0.004; // running estimate of room noise
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
  function maybeNudge() {
    if (stopped || busy || muted || speaking || nudges >= quietGaps.length) return;
    if (performance.now() - lastActive < quietGaps[nudges] * 1000 * (0.75 + Math.random() * 0.5)) return;
    nudges += 1;
    void runTurn({ nudge: nudges });
  }

  async function post(body, onEvent) {
    const res = await fetch("/api/chat/turn", {
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
          o.onLine("avatar", ev.text);
          const blob = b64ToBlob(ev.audio, "audio/wav");
          chain = chain.then(async () => {
            if (stopped) return null;
            voiceOut = true;
            try {
              return await o.playClip(blob, ev.video_url || "");
            } finally {
              voiceOut = false;
              voiceOutUntil = performance.now() + 1100; // its own voice fades out of the room
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
    o.onStatus("Thinking…");
    let failed = "";
    try {
      await post(body, (ev) => {
        if (ev.type === "superseded") superseded = true;
        if (ev.type === "heard") {
          engaged = true;
          nudges = 0; // you really said something
          o.onLine("you", ev.text);
        }
        if (ev.type === "error") failed = ev.message;
      });
    } catch (e) {
      failed = e.message || String(e);
    }
    if (superseded) {
      api.stop();
      o.onSuperseded?.();
      return;
    }
    if (failed) o.onStatus(`Problem: ${failed}`);
    resumeAt = performance.now() + 600; // let the room go quiet before listening again
    busy = false;
    if (engaged || body.nudge || body.say) lastActive = performance.now(); // room noise that was ignored does not restart the silence clock
    if (pending.length && !failed && !stopped && !muted) {
      const got = pending;
      pending = [];
      sendHeard(got); // you kept talking while it was busy: answer that now
      return;
    }
    pending = [];
    if (!failed && !stopped) o.onStatus(muted ? "Mic muted." : "Listening… just talk.");
  }

  function sendHeard(got) {
    const wav = encodeWav16k(got, ctx.sampleRate);
    blobToBase64(wav).then((b64) => runTurn({ audio_base64: b64 }));
  }

  function onAudio(e) {
    const x = e.inputBuffer.getChannelData(0);
    let sum = 0;
    for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
    const rms = Math.sqrt(sum / x.length);
    const ms = (x.length / ctx.sampleRate) * 1000;
    const ownVoice = voiceOut || performance.now() < voiceOutUntil;
    const threshold = Math.max(0.01, floor * 3.2);
    o.onLevel(rms, threshold);
    if (ownVoice) {
      // its own voice is coming out of the speakers: record nothing (it would hear itself and answer itself)
      speaking = false;
      chunks = [];
      preroll = [];
      loud = 0;
      return;
    }
    if (muted || (!busy && performance.now() < resumeAt)) {
      speaking = false;
      chunks = [];
      loud = 0;
      return;
    }
    if (!speaking) {
      if (rms < threshold) floor = floor * 0.97 + rms * 0.03; // learn the room only while it is quiet
      preroll.push(new Float32Array(x));
      if (preroll.length > 4) preroll.shift();
      loud = rms > threshold ? loud + 1 : 0;
      if (loud >= 2) {
        speaking = true;
        chunks = [...preroll];
        speechMs = 0;
        quietMs = 0;
        if (!busy) o.onStatus("Hearing you…");
      }
      return;
    }
    chunks.push(new Float32Array(x));
    speechMs += ms;
    quietMs = rms < threshold * 0.8 ? quietMs + ms : 0;
    if (quietMs >= 900 || speechMs > 25000) {
      const got = chunks;
      speaking = false;
      chunks = [];
      preroll = [];
      loud = 0;
      if (speechMs - quietMs < 350) {
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
      o.onStatus("Listening… just talk.");
      nudgeTimer = setInterval(maybeNudge, 1000);
    },
    say: (text) => runTurn({ say: text }),
    /** Restart the silence clock (after the greeting, say). */
    touch() {
      lastActive = performance.now();
      nudges = 0;
    },
    sendText: (text) => runTurn({ text }),
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
