import { buildMeshFromScans, createMeshPuppet, meshFromLandmarks } from "/static/face-mesh-avatar.js?v=20";
import { createVoiceChat } from "/static/voice-chat.js?v=2";
import { loadCinematic } from "/static/cinematic-clips.js?v=2";


/**
 * @param {HTMLElement} stageEl .avatar-stage
 * @param {object} profile profile_summary from API
 * @returns {() => void} teardown
 */
export function mountAvatarPlayer(stageEl, profile, hooks = {}) {
  return mountLiveAnimatedPlayer(stageEl, profile, hooks);
}

/** Real-time mouth, blink, and head motion driven by voice (3D stitched skin when available). */
function mountLiveAnimatedPlayer(stageEl, profile, hooks = {}) {
  const name = profile.display_name || profile.canonical_name || "Avatar";
  const frames = (profile.scan_frames?.length ? profile.scan_frames : profile.scan_previews) || [];
  const fallbackFace = profile.avatar_face_url || frames[0]?.url || "";

  stageEl.innerHTML = `
    <div class="avatar-figure mesh has-face" id="avatar">
      <canvas class="avatar-canvas" id="avatar-canvas"></canvas>
    </div>
    <h2>${escapeHtml(name)}</h2>
    <p id="avatar-status" class="avatar-note muted">Building your face from the scan…</p>
    <div class="chat-meter" title="Your microphone level"><div class="chat-meter-bar" id="chat-meter-bar"></div><div class="chat-meter-thr" id="chat-meter-thr"></div></div>
    <div class="chat-log" id="chat-log" aria-live="polite"></div>
    <form class="chat-type" id="chat-type" autocomplete="off">
      <input id="chat-input" type="text" placeholder="…or type a message" />
      <button type="submit" class="btn-ghost">Send</button>
    </form>
    <audio id="avatar-audio" preload="auto"></audio>
    <div class="avatar-actions btn-row" style="justify-content: center; flex-wrap: wrap; gap: 0.5rem;">
      <button type="button" class="btn-primary" id="chat-mute" disabled>Mute mic</button>
      <button type="button" class="btn-ghost" id="back-profile">Back to profile</button>
    </div>
  `;

  const audio = stageEl.querySelector("#avatar-audio");
  const canvas = stageEl.querySelector("#avatar-canvas");
  const statusEl = stageEl.querySelector("#avatar-status");
  const muteBtn = stageEl.querySelector("#chat-mute");
  const logEl = stageEl.querySelector("#chat-log");
  const meterBar = stageEl.querySelector("#chat-meter-bar");
  const meterThr = stageEl.querySelector("#chat-meter-thr");
  let chat = null;
  let cinema = null; // pre-rendered full-quality clips (greeting / thinking), when the library exists

  let puppet = null;
  let rafId = 0;
  let destroyed = false;
  let playing = false;
  let analysis = null; // per-20ms features of the recording, computed offline (no AudioContext needed to play)

  // Smoothed mouth state.
  const mouth = { open: 0, wide: 0, env: 0, peak: 0.05 };
  // Blink scheduling.
  const blink = { next: 1.5, start: -1, dur: 0.19, double: false };

  const HOP_S = 0.02;
  const LEAD_S = 0.05; // mouth leads the sound slightly, like real speech

  /** Decode the recording and measure loudness + low/mid/high balance every 20 ms. */
  async function analyzeVoice(buf) {
    const Off = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    const ab = await new Off(1, 1, 44100).decodeAudioData(buf);
    const sr = ab.sampleRate;
    const x = ab.getChannelData(0);
    const N = 1024;
    const hop = Math.round(sr * HOP_S);
    const count = Math.max(1, Math.floor((x.length - N) / hop));
    const re = new Float32Array(N);
    const im = new Float32Array(N);
    const win = new Float32Array(N).map((_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1)));
    const binHz = sr / N;
    const bandEnergy = (lo, hi) => {
      let e = 0;
      for (let k = Math.max(1, Math.floor(lo / binHz)); k <= Math.min(N / 2 - 1, Math.ceil(hi / binHz)); k++) e += re[k] * re[k] + im[k] * im[k];
      return e;
    };
    const out = { rms: new Float32Array(count), lowF: new Float32Array(count), midF: new Float32Array(count), highF: new Float32Array(count) };
    for (let f = 0; f < count; f++) {
      const off = f * hop;
      let sum = 0;
      for (let i = 0; i < N; i++) {
        const v = x[off + i] || 0;
        sum += v * v;
        re[i] = v * win[i];
        im[i] = 0;
      }
      out.rms[f] = Math.sqrt(sum / N);
      fft(re, im);
      const lo = bandEnergy(100, 700);
      const mi = bandEnergy(700, 2600);
      const hi = bandEnergy(3500, 9000);
      const tot = lo + mi + hi + 1e-12;
      out.lowF[f] = lo / tot;
      out.midF[f] = mi / tot;
      out.highF[f] = hi / tot;
    }
    const sorted = Array.from(out.rms).sort((a, b) => a - b);
    out.peak = Math.max(0.02, sorted[Math.floor(sorted.length * 0.95)] || 0.05);
    return out;
  }

  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (-2 * Math.PI) / len;
      const wr = Math.cos(ang);
      const wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1;
        let ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const ur = re[i + k];
          const ui = im[i + k];
          const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
          const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
          re[i + k] = ur + vr;
          im[i + k] = ui + vi;
          re[i + k + len / 2] = ur - vr;
          im[i + k + len / 2] = ui - vi;
          const nr = cr * wr - ci * wi;
          ci = cr * wi + ci * wr;
          cr = nr;
        }
      }
    }
  }

  function audioFeatures() {
    if (!playing || !audio || audio.paused) {
      mouth.open += (0 - mouth.open) * 0.35;
      mouth.wide += (0 - mouth.wide) * 0.2;
      mouth.env += (0 - mouth.env) * 0.1;
      return;
    }
    let openT;
    let wideT = 0;
    if (analysis) {
      const i = clamp(Math.floor((audio.currentTime + LEAD_S) / HOP_S), 0, analysis.rms.length - 1);
      // average 3 frames so single-frame spikes don't flicker the jaw
      const i0 = Math.max(0, i - 1);
      const i1 = Math.min(analysis.rms.length - 1, i + 1);
      const rms = (analysis.rms[i0] + analysis.rms[i] + analysis.rms[i1]) / 3;
      const lowF = analysis.lowF[i];
      const midF = analysis.midF[i];
      const highF = analysis.highF[i];
      openT = rms < Math.max(0.004, analysis.peak * 0.07) ? 0 : Math.pow(clamp(rms / (analysis.peak * 0.85), 0, 1), 0.7);
      wideT = clamp((midF - lowF) * 1.2, -0.5, 0.7); // "ee" spreads, "oo/aw" rounds
      if (highF > 0.45) {
        // s / sh / f: teeth nearly together, lips slightly spread
        openT *= 0.45;
        wideT = Math.max(wideT, 0.3);
      }
    } else {
      // Analysis unavailable: plausible jaw motion while the audio plays.
      const t = audio.currentTime;
      openT = clamp(0.35 + 0.3 * Math.sin(t * 11) * Math.sin(t * 3.1) + 0.15 * Math.sin(t * 23), 0, 1);
    }
    mouth.open += (openT - mouth.open) * (openT > mouth.open ? 0.65 : 0.3);
    mouth.wide += (wideT * Math.min(1, openT * 2 + 0.2) - mouth.wide) * 0.35;
    mouth.env += (openT - mouth.env) * 0.08;
  }

  function stepBlink(t) {
    if (blink.start < 0 && t >= blink.next) {
      blink.start = t;
      blink.dur = 0.16 + Math.random() * 0.06;
    }
    if (blink.start < 0) return 0;
    const p = (t - blink.start) / blink.dur;
    if (p >= 1) {
      blink.start = -1;
      blink.double = !blink.double && Math.random() < 0.15;
      blink.next = t + (blink.double ? 0.18 : 2.2 + Math.random() * 3.8);
      return 0;
    }
    // quick close (35%), slower open
    return p < 0.35 ? p / 0.35 : 1 - (p - 0.35) / 0.65;
  }

  function frame() {
    if (destroyed || !puppet) return;
    const t = performance.now() / 1000;
    audioFeatures();
    const s = mouth.env;
    const b = stepBlink(t);
    const pose = {
      open: mouth.open,
      wide: mouth.wide,
      press: mouth.open < 0.05 ? 0.6 : 0, // lips close between words
      blinkL: b,
      blinkR: b * (0.96 + 0.04 * Math.sin(t * 7)),
      brow: 0.15 * Math.sin(t * 0.7) + 0.55 * s,
      // Head turns in real 3D: slow look-around, livelier while talking.
      // Kept small on purpose: this warps ONE flat photo over a rough depth map, so anything past a few degrees bends the jaw and nose.
      yaw: 0.045 * Math.sin(t * 0.43) + 0.015 * Math.sin(t * 1.1 + 1) + s * 0.03 * Math.sin(t * 2.3),
      pitch: 0.012 * Math.sin(t * 0.61 + 0.7) + 0.005 * Math.sin(t * 1.3) + s * 0.02 * Math.sin(t * 3.1 + 0.5) - mouth.open * 0.01,
      roll: 0.012 * Math.sin(t * 0.37) + s * 0.015 * Math.sin(t * 1.7),
      dx: 3.0 * Math.sin(t * 0.5),
      dy: 2.0 * Math.sin(t * 0.8) + s * 2,
      zoom: 1.0 + 0.004 * Math.sin(t * 1.3),
    };
    try {
      puppet.render(pose);
    } catch (e) {
      console.error("render failed", e);
      return;
    }
    rafId = requestAnimationFrame(frame);
  }

  /** Play one reply clip; the mouth follows it (loudness + vowel colour analysed up front). */
  async function playClip(blob) {
    if (destroyed) return;
    clearTimeout(thinkingTimer);
    await cinema?.whenIdle();
    try {
      analysis = await analyzeVoice(await blob.arrayBuffer());
    } catch (e) {
      console.warn("clip analysis failed (using fallback jaw motion):", e);
      analysis = null;
    }
    audio.src = URL.createObjectURL(blob);
    playing = true;
    statusEl.textContent = "Speaking…";
    await new Promise((resolve) => {
      const done = () => {
        audio.removeEventListener("ended", done);
        audio.removeEventListener("error", done);
        resolve();
      };
      audio.addEventListener("ended", done);
      audio.addEventListener("error", done);
      audio.play().catch((err) => {
        console.error(err);
        statusEl.textContent = `Could not play audio (${err?.name || "error"}) — click the page once, then talk again.`;
        done();
      });
    });
    playing = false;
    URL.revokeObjectURL(audio.src);
  }

  function addLine(who, text) {
    const row = document.createElement("p");
    row.className = `chat-line ${who}`;
    row.textContent = (who === "you" ? "You: " : `${name}: `) + text;
    logEl.appendChild(row);
    logEl.scrollTop = logEl.scrollHeight;
  }

  let chatBrainFoot = "";

  /** Play a full-quality clip over the avatar; the mic is muted meanwhile so the chat does not hear it. */
  async function playCinematic(kind) {
    if (!cinema || destroyed) return false;
    const wasMuted = chat ? chat.muted : false;
    chat?.setMuted(true);
    try {
      return await cinema.play(kind);
    } finally {
      if (!wasMuted && !destroyed) chat?.setMuted(false);
    }
  }
  let thinkingTimer = 0;
  function onChatStatus(t) {
    if (destroyed) return;
    statusEl.textContent = t + chatBrainFoot;
    clearTimeout(thinkingTimer);
    // The reply takes a few seconds: look thoughtful for a moment instead of freezing.
    if (cinema && /^Thinking/.test(t)) thinkingTimer = setTimeout(() => void playCinematic("thinking"), 600);
  }

  async function startChat() {
    statusEl.textContent = "Loading the voice and speech models (the first time takes a minute)…";
    try {
      const r = await fetch("/api/chat/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ canonical_name: profile.canonical_name }),
      });
      const j = await r.json();
      if (j.status !== "ok") throw new Error(j.message || "chat could not start");
      if (destroyed) return;
      chatBrainFoot =
        j.llm_model && j.llm_mode ? ` (brain: ${j.llm_model}, ${j.llm_mode})` : "";
      chat = createVoiceChat({
        name: profile.canonical_name,
        token: j.token,
        playClip,
        onSuperseded: () => {
          audio.pause();
          statusEl.textContent = "Chat moved to another window — close the extra avatar tabs, then reload this one.";
          muteBtn.disabled = true;
        },
        onStatus: onChatStatus,
        onLevel: (v, thr) => {
          meterBar.style.width = `${Math.min(100, v * 600)}%`;
          meterThr.style.left = `${Math.min(100, thr * 600)}%`;
        },
        onLine: addLine,
      });
      await chat.start();
      muteBtn.disabled = false;
      cinema = await loadCinematic(stageEl, profile, { onNote: (m) => !destroyed && (statusEl.textContent = m) }).catch(() => null);
      if (!(cinema && (await playCinematic("greeting")))) await chat.say(`Hello, I'm ${j.display_name || name}.`);
    } catch (e) {
      console.error(e);
      statusEl.textContent =
        e?.name === "NotAllowedError" || e?.name === "NotFoundError"
          ? "I can't use the microphone — allow it in the browser's address bar, then reload."
          : `Chat problem: ${e.message || e}`;
    }
  }

  stageEl.querySelector("#back-profile")?.addEventListener("click", () => hooks.onBackProfile?.());

  muteBtn?.addEventListener("click", () => {
    if (!chat) return;
    chat.setMuted(!chat.muted);
    muteBtn.textContent = chat.muted ? "Unmute mic" : "Mute mic";
  });
  stageEl.querySelector("#chat-type")?.addEventListener("submit", (e) => {
    e.preventDefault();
    const input = stageEl.querySelector("#chat-input");
    const t = input.value.trim();
    if (t && chat) {
      input.value = "";
      void chat.sendText(t);
    }
  });

  // Build the puppet from the scan photos.
  (async () => {
    try {
      if (!frames.length) throw new Error("No scan photos yet — do the face scan first.");
      // Preferred: landmarks found on the server (works in every browser). Fallback: in-browser detection.
      let mesh = null;
      try {
        statusEl.textContent = "Finding your face…";
        const r = await fetch("/api/avatar/landmarks", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ canonical_name: profile.canonical_name }),
        });
        const j = await r.json();
        if (j.status === "ok") mesh = await meshFromLandmarks(j.url, j.landmarks);
        else console.warn("server landmarks unavailable:", j.message);
      } catch (e) {
        console.warn("server landmarks failed:", e);
      }
      if (!mesh) mesh = await buildMeshFromScans(frames, (n, total, msg) => {
        statusEl.textContent = msg ? `${msg} (photo ${n}/${total})` : `Finding the most frontal scan photo… ${n}/${total}`;
      });
      if (destroyed) return;
      puppet = createMeshPuppet(canvas, mesh);
      rafId = requestAnimationFrame(frame);
      void startChat();
    } catch (err) {
      console.error(err);
      statusEl.textContent = `Could not build the face mesh: ${err.message}`;
      if (fallbackFace) {
        const fig = stageEl.querySelector("#avatar");
        fig.innerHTML = `<img class="avatar-face" src="${escapeAttr(fallbackFace)}" alt="" />`;
        // No mesh to animate: give the still image a small speaking bob so you can see it talking.
        const img = fig.querySelector("img");
        const bob = () => {
          if (destroyed) return;
          audioFeatures();
          const t = performance.now() / 1000;
          img.style.transform = `translateY(${(-mouth.open * 4).toFixed(2)}px) scale(${(1 + mouth.open * 0.03).toFixed(4)}) rotate(${(Math.sin(t * 0.7) * 0.6 + mouth.env * Math.sin(t * 3) * 1.2).toFixed(2)}deg)`;
          rafId = requestAnimationFrame(bob);
        };
        rafId = requestAnimationFrame(bob);
      }
      void startChat(); // the voice conversation does not need a face mesh
    }
  })();

  return () => {
    destroyed = true;
    cancelAnimationFrame(rafId);
    audio?.pause();
    chat?.stop();
    clearTimeout(thinkingTimer);
    cinema?.dispose();
    try {
      puppet?.dispose?.();
    } catch {
      /* ignore */
    }
  };
}

function clamp(v, a, b) {
  return Math.max(a, Math.min(b, v));
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function escapeAttr(s) {
  return escapeHtml(s).replace(/"/g, "&quot;");
}
