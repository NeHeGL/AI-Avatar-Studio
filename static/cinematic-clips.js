/**
 * Photo-real layer drawn over the live avatar: pre-rendered "cinematic" clips (greeting, thinking, reactions,
 * closing) and live-rendered reply videos, all from the same wide photo so the framing never jumps.
 *
 * ml/cinematic_build.py renders the clips into data/media/<user>/cinematic/ (manifest.json + the spoken line as
 * src/<id>.wav). Replies come from the local Ditto render service via the chat server (video_url per sentence).
 * With no library the module returns null and everything falls back silently to the animated face.
 *
 * Two <video> layers cross-fade, so the picture never drops out between one clip and the next. Pictures play
 * MUTED (browsers always allow that); voices play through normal audio elements, the same way the chat's own
 * voice does, so nothing depends on the browser's video-codec or autoplay-with-sound rules.
 *
 *   const cinema = await loadCinematic(stageEl, profile, { onNote });   // null when there is no library
 *   await cinema?.play("greeting");                                    // fixed clip; resolves when finished
 *   await cinema?.playReply(videoUrl, wavBlob);                         // live reply sentence
 *   await cinema?.whenIdle();   cinema?.release();   cinema?.dispose();
 */
export async function loadCinematic(stageEl, profile, opts = {}) {
  const note = (m) => {
    try {
      console.warn("[cinematic]", m);
      opts.onNote?.(m);
    } catch {
      /* ignore */
    }
  };
  const frames = (profile.scan_frames?.length ? profile.scan_frames : profile.scan_previews) || [];
  const sample = profile.avatar_face_url || frames[0]?.url || "";
  const base = (sample.match(/^(.*\/media\/[^/]+)\//) || [])[1];
  if (!base) return null;

  // The server prepares the person's photo (from whatever scan photos exist) and says if a clip library exists.
  let prep = null;
  try {
    const r = await fetch("/api/photoreal/prepare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ canonical_name: profile.canonical_name }),
    });
    prep = await r.json();
  } catch {
    return null;
  }
  if (!prep?.has_source) return null;
  let manifest = { clips: [] };
  if (prep.has_clips) {
    try {
      const r = await fetch(`${base}/cinematic/manifest.json`, { cache: "no-store" });
      if (r.ok) manifest = await r.json();
    } catch {
      /* no clips: the idle face and the reply videos still work */
    }
  }
  const clips = (manifest.clips || []).filter((c) => c.video);

  const figure = stageEl.querySelector("#avatar");
  if (!figure) return null;
  figure.style.position = "relative";

  // Everything that moves lives in `fg`. When a cut-out of the person exists, `fg` is masked to it and a background sits behind,
  // so the background can be replaced (or blurred) without touching the animation.
  const bg = document.createElement("div");
  bg.style.cssText = "position:absolute;inset:0;border-radius:inherit;z-index:0;pointer-events:none;";
  figure.appendChild(bg);
  const fg = document.createElement("div");
  fg.style.cssText = "position:absolute;inset:0;border-radius:inherit;z-index:1;overflow:hidden;";
  figure.appendChild(fg);
  const matteUrl = `${base}/cinematic/matte.png`;
  const sourceUrl = `${base}/cinematic/source.jpg`;
  const BG_KEY = `avatar-bg2-${base.split("/").pop()}`;
  let matteReady = false;
  const bokeh = (colors) =>
    colors
      .map(([x, y, r, c]) => `radial-gradient(circle at ${x}% ${y}%, ${c} 0, ${c} ${(r * 0.25).toFixed(1)}%, transparent ${(r * 0.8).toFixed(1)}%)`)
      .join(",");
  const BACKGROUNDS = {
    original: { label: "Original" },
    studio: { label: "Studio", css: "radial-gradient(ellipse at 50% 30%, #3b4252 0%, #232838 45%, #11141c 100%)" },
    blur: { label: "Blurred room", css: `url(${sourceUrl})`, filter: "blur(7px) brightness(.9) saturate(1.1)" },
    night: {
      label: "City night",
      css: `${bokeh([[18, 30, 7, "rgba(255,190,90,.55)"], [78, 22, 9, "rgba(120,180,255,.5)"], [60, 62, 6, "rgba(255,120,150,.45)"], [30, 70, 8, "rgba(140,255,200,.35)"], [88, 70, 5, "rgba(255,230,150,.5)"], [8, 78, 5, "rgba(200,140,255,.4)"]])},linear-gradient(180deg,#0b1026 0%,#1a1f45 60%,#2b2350 100%)`,
    },
    sunset: { label: "Sunset", css: "radial-gradient(circle at 50% 78%, #ffd27a 0%, #ff8a5c 28%, #c4457a 55%, #3a2a6b 100%)" },
    sky: { label: "Sky", css: `${bokeh([[22, 24, 14, "rgba(255,255,255,.7)"], [70, 18, 10, "rgba(255,255,255,.6)"], [80, 40, 16, "rgba(255,255,255,.45)"]])},linear-gradient(180deg,#5aa9f0 0%,#a8d4ff 55%,#e9f4ff 100%)` },
    forest: { label: "Forest", css: `${bokeh([[20, 25, 9, "rgba(190,255,160,.35)"], [65, 15, 12, "rgba(220,255,180,.3)"], [85, 55, 8, "rgba(160,255,190,.3)"], [35, 60, 11, "rgba(210,255,150,.25)"]])},linear-gradient(170deg,#2f6b45 0%,#1b4a33 60%,#0f2e22 100%)` },
    warm: { label: "Warm office", css: `${bokeh([[15, 35, 8, "rgba(255,225,160,.5)"], [80, 30, 11, "rgba(255,210,140,.4)"], [55, 70, 7, "rgba(255,240,200,.35)"]])},linear-gradient(160deg,#e3b58a 0%,#a8714b 60%,#6b3f27 100%)` },
  };
  function applyBackground(key) {
    let choice = BACKGROUNDS[key];
    if (key === "custom") {
      let image = "";
      try {
        image = localStorage.getItem(`${BG_KEY}-img`) || "";
      } catch {
        /* ignore */
      }
      choice = image ? { css: `url(${image})` } : BACKGROUNDS.original;
    }
    const on = Boolean(matteReady && choice && choice.css);
    fg.style.webkitMaskImage = fg.style.maskImage = on ? `url(${matteUrl})` : "none";
    fg.style.webkitMaskSize = fg.style.maskSize = "100% 100%";
    bg.style.background = on ? choice.css : "none";
    bg.style.backgroundSize = "cover";
    bg.style.backgroundPosition = "center";
    bg.style.filter = on && choice.filter ? choice.filter : "none";
    bg.style.transform = on && choice.filter ? "scale(1.08)" : "none";
  }
  const rememberBackground = (key) => {
    try {
      localStorage.setItem(BG_KEY, key); // only what the person picked is remembered
    } catch {
      /* ignore */
    }
  };
  let bgKey = "original"; // the default look
  try {
    bgKey = localStorage.getItem(BG_KEY) || "original";
  } catch {
    /* ignore */
  }
  let matteTries = 0;
  let matteTimer = 0;
  let disposed = false;
  let pickerEl = null;
  // the cut-out is made by the live-face service the first time it sees a person; look for it until it shows up
  const lookForMatte = () => {
    const probe = new Image();
    probe.onload = () => {
      matteReady = true;
      applyBackground(bgKey);
      mountPicker();
    };
    probe.onerror = () => {
      if (++matteTries < 40 && !disposed) matteTimer = setTimeout(lookForMatte, 2500);
    };
    probe.src = `${matteUrl}?t=${Date.now()}`;
  };
  function mountPicker() {
    if (pickerEl || disposed) return;
    pickerEl = document.createElement("div");
    pickerEl.style.cssText = "display:flex;flex-wrap:wrap;gap:.4rem;justify-content:center;margin:.2rem 0 .6rem;align-items:center;";
    const mk = (key, label) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "btn-ghost";
      b.textContent = label;
      b.style.cssText = "padding:.2rem .6rem;font-size:.8rem;";
      b.addEventListener("click", () => {
        bgKey = key;
        applyBackground(key);
        rememberBackground(key);
      });
      return b;
    };
    for (const [k, v] of Object.entries(BACKGROUNDS)) pickerEl.appendChild(mk(k, v.label));
    const file = document.createElement("input");
    file.type = "file";
    file.accept = "image/*";
    file.style.display = "none";
    file.addEventListener("change", () => {
      const f = file.files && file.files[0];
      if (!f) return;
      const img = new Image();
      img.onload = () => {
        const c = document.createElement("canvas");
        const k = Math.min(1, 900 / Math.max(img.width, img.height));
        c.width = Math.round(img.width * k);
        c.height = Math.round(img.height * k);
        c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
        try {
          localStorage.setItem(`${BG_KEY}-img`, c.toDataURL("image/jpeg", 0.85)); // stays in this browser only
        } catch {
          /* too big: just use it this session */
        }
        bgKey = "custom";
        applyBackground("custom");
        rememberBackground("custom");
      };
      img.src = URL.createObjectURL(f);
    });
    const pick = mk("pick", "Your image…");
    pick.onclick = () => file.click();
    pickerEl.append(pick, file);
    figure.insertAdjacentElement("afterend", pickerEl);
  }
  lookForMatte();

  // Base layer: the same prepared photo every clip and reply is made from, so size and framing always match.
  const baseStill = new Image();
  baseStill.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;border-radius:inherit;z-index:0;pointer-events:none;";
  baseStill.src = `${base}/cinematic/source.jpg`;
  fg.appendChild(baseStill);
  const rawStill = stageEl.querySelector("#avatar-still");
  if (rawStill) rawStill.style.display = "none";

  const makeVideo = () => {
    const v = document.createElement("video");
    v.playsInline = true;
    v.muted = true; // picture only; voices come from audio elements
    v.preload = "auto";
    v.style.cssText =
      "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;border-radius:inherit;opacity:0;pointer-events:none;transition:opacity .2s ease;";
    fg.appendChild(v);
    return v;
  };
  const layers = [makeVideo(), makeVideo()];

  // ---- Bank of pre-made waiting animations. Every clip starts and ends on the same rest frame, so they can follow each other
  // (or a pause on the rest frame) without a visible join, and they cost no computing while a reply is being prepared. ----
  const bankBase = `${base}/cinematic/idle_bank`;
  const bankVideo = document.createElement("video");
  bankVideo.playsInline = true;
  bankVideo.muted = true;
  bankVideo.preload = "auto";
  bankVideo.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;border-radius:inherit;opacity:0;pointer-events:none;z-index:1;";
  fg.appendChild(bankVideo);
  let bank = [];
  let bankPlaying = false;
  let mainActive = false; // a greeting / reply is on screen (or about to be)
  let thinkUntil = 0; // while a reply is being prepared: only thoughtful animations
  let liveWanted = false;
  let bankTimer = 0;
  let resumeTimer = 0;
  let bankLoadTimer = 0;
  let lastBank = "";
  let bankWasOn = false;
  let restWaiters = [];
  const bankOn = () => bank.length >= 6;
  const showLive = () => !bankOn() || liveWanted;
  const resolveRest = () => {
    const w = restWaiters;
    restWaiters = [];
    for (const f of w) f();
  };
  const whenRest = (maxMs = 4500) =>
    bankPlaying ? new Promise((res) => {
      restWaiters.push(res);
      setTimeout(res, maxMs);
    }) : Promise.resolve();

  function pickBank() {
    const thinking = Date.now() < thinkUntil;
    const pool = bank.filter((c) => c.id !== lastBank && (!thinking || c.think || c.kind === "breathe"));
    if (!pool.length) return null;
    const weight = (c) => (c.kind === "breathe" ? (thinking ? 2 : 6) : c.kind === "yawn" ? 0.4 : c.think && thinking ? 3 : 1.5);
    let r = Math.random() * pool.reduce((a, c) => a + weight(c), 0);
    for (const c of pool) {
      r -= weight(c);
      if (r <= 0) return c;
    }
    return pool[0];
  }
  async function playBank(c) {
    bankPlaying = true;
    lastBank = c.id;
    try {
      bankVideo.src = `${bankBase}/${c.file}`;
      bankVideo.currentTime = 0;
      await bankVideo.play();
      bankVideo.style.opacity = "1";
      await new Promise((res) => {
        bankVideo.onended = res;
        bankVideo.onerror = res;
        setTimeout(res, (c.dur + 1) * 1000);
      });
    } catch {
      /* skip this one */
    }
    bankPlaying = false;
    resolveRest();
  }
  function tickBank() {
    clearTimeout(bankTimer);
    if (disposed || !bankOn() || mainActive || bankPlaying) return;
    const c = pickBank();
    if (!c) return;
    playBank(c).then(() => {
      if (!mainActive && !disposed) bankTimer = setTimeout(tickBank, Date.now() < thinkUntil ? 60 : 150 + Math.random() * 900);
    });
  }
  function setLive(on) {
    liveWanted = on;
    try {
      fetch(`${IDLE}/live?on=${on ? 1 : 0}`, { method: "POST", mode: "no-cors" }).catch(() => {});
    } catch {
      /* ignore */
    }
    if (idleImg) idleImg.style.opacity = idleOk && showLive() ? "1" : "0";
  }
  async function loadBank() {
    try {
      const r = await fetch(`${bankBase}/manifest.json`, { cache: "no-store" });
      if (r.ok) bank = (await r.json()).clips || [];
    } catch {
      /* no bank yet: the live face does the waiting */
    }
    if (bankOn() && !bankWasOn) {
      bankWasOn = true;
      baseStill.src = `${bankBase}/rest.jpg`; // the exact frame every animation starts and ends on
      setLive(false);
      tickBank();
    }
    if (!disposed) bankLoadTimer = setTimeout(loadBank, bank.length < 40 ? 8000 : 60000);
  }
  /** After a greeting or reply: let the live face carry on from the last frame, then settle back to the pre-made animations. */
  function backToBank() {
    clearTimeout(resumeTimer);
    resumeTimer = setTimeout(() => {
      mainActive = false;
      if (bankOn()) {
        setLive(false);
        bankVideo.style.opacity = "0";
        tickBank();
      }
    }, 3400);
  }
  // the live-face settings below are only read after this first network wait, so starting here is safe
  void loadBank();
  let front = -1; // index of the layer currently showing, -1 = none yet
  let fadeTimer = 0;

  // Live idle layer: the same photo, alive in real time (blinks, breathing, slow drift, yawns). It sits under the videos
  // and is what you see whenever nothing is playing, so the face never freezes.
  const IDLE = opts.idleUrl || "http://localhost:8770";
  const slug = base.split("/").pop();
  let idleOk = false;
  let idleImg = null;
  let idlePoll = 0;
  let lastFrames = -1;
  let stale = 0;
  let frameNo = 0;
  let lastFrameAt = 0;
  let stopFrames = () => {};
  const streamUrl = () => `${IDLE}/stream?u=${Date.now()}`;
  function startIdle() {
    try {
      fetch(`${IDLE}/use?user=${encodeURIComponent(slug)}`, { method: "POST", mode: "no-cors" }).catch(() => {});
    } catch {
      return;
    }
    const img = new Image();
    img.style.cssText =
      "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;border-radius:inherit;z-index:0;opacity:0;transition:opacity .45s ease;pointer-events:none;";
    fg.appendChild(img);
    idleImg = img;
    // Pull the newest frame again and again (a live stream is flaky in some browsers).
    let alive = true;
    stopFrames = () => (alive = false);
    const next = (delay) => setTimeout(() => alive && (img.src = `${IDLE}/frame.jpg?n=${++frameNo}`), delay);
    img.onload = () => {
      lastFrameAt = Date.now();
      next(30);
    };
    img.onerror = () => next(600);
    next(0);
    const poll = async () => {
      try {
        const r = await fetch(`${IDLE}/health`, { cache: "no-store" });
        const j = await r.json();
        idleOk = Boolean(j.ok && j.user === slug && j.frames > 0);
        if (idleOk) {
          if (Date.now() - lastFrameAt > 3000) img.src = `${IDLE}/frame.jpg?n=${++frameNo}`; // frames stopped arriving: kick it
        }
      } catch {
        idleOk = false;
      }
      img.style.opacity = idleOk && showLive() ? "1" : "0";
    };
    poll();
    idlePoll = setInterval(poll, 1500);
  }
  startIdle();

  /** Tell the live face what is on screen right now, so it carries on from that pose and expression (no jump). */
  async function seedFromFront() {
    const v = frontVideo();
    if (!idleOk || !v || !v.videoWidth) return false;
    try {
      const c = document.createElement("canvas");
      const k = Math.min(1, 512 / v.videoWidth);
      c.width = Math.round(v.videoWidth * k);
      c.height = Math.round(v.videoHeight * k);
      c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
      const blob = await new Promise((res) => c.toBlob(res, "image/jpeg", 0.9));
      const r = await fetch(`${IDLE}/seed`, { method: "POST", body: blob });
      return Boolean((await r.json()).ok);
    } catch {
      return false;
    }
  }

  /** on=true: the live face goes still and neutral (a reply is being prepared) so the reply video fades in from a match. */
  function settle(on) {
    if (bankOn()) {
      thinkUntil = on ? Date.now() + 20000 : 0;
      if (on) tickBank();
      return;
    }
    if (!idleOk) return;
    fetch(`${IDLE}/settle?on=${on ? 1 : 0}`, { method: "POST", mode: "no-cors" }).catch(() => {});
  }

  /** Start `url` on the hidden layer, then fade it in over the one showing. Resolves with that video element. */
  async function startVideo(url) {
    const next = layers[front < 0 ? 0 : 1 - front];
    const old = front < 0 ? null : layers[front];
    clearTimeout(fadeTimer);
    next.style.transition = "opacity .15s ease";
    next.style.opacity = "0";
    next.style.zIndex = "2";
    next.src = url;
    next.currentTime = 0;
    await next.play(); // resolves once the first frame is playing, so the fade never shows a blank picture
    next.style.opacity = "1";
    setTimeout(() => (bankVideo.style.opacity = "0"), 220);
    if (old) {
      old.style.zIndex = "1";
      setTimeout(() => {
        try {
          old.pause();
        } catch {
          /* ignore */
        }
        old.style.opacity = "0";
      }, 260);
    }
    front = front < 0 ? 0 : 1 - front;
    return next;
  }
  const frontVideo = () => (front < 0 ? null : layers[front]);

  let current = null; // promise of the running clip
  const lastByKind = {};

  function pick(kind) {
    const pool = clips.filter((c) => c.kind === kind);
    if (!pool.length) return null;
    const fresh = pool.filter((c) => c.id !== lastByKind[kind]);
    const list = fresh.length ? fresh : pool;
    const c = list[Math.floor(Math.random() * list.length)];
    lastByKind[kind] = c.id;
    return c;
  }

  /** Common run-loop: show a video, play a voice, resolve shortly after the voice ends. The last frame stays up. */
  function run({ videoUrl, voiceUrl, maxMs, label }) {
    mainActive = true;
    clearTimeout(bankTimer);
    clearTimeout(resumeTimer);
    thinkUntil = 0;
    current = new Promise((resolve) => {
      let finished = false;
      let tail = 0;
      const voice = new Audio(voiceUrl);
      voice.preload = "auto";
      const finish = (ok) => {
        if (finished) return;
        finished = true;
        clearTimeout(tail);
        try {
          voice.pause();
          frontVideo()?.pause(); // hold the last frame (the picture of you) rather than dropping back to the animated face
        } catch {
          /* ignore */
        }
        if (voiceUrl.startsWith("blob:")) URL.revokeObjectURL(voiceUrl);
        current = null;
        if (idleOk) {
          clearTimeout(fadeTimer);
          // hold the last frame for a moment, tell the live face what it looks like, then fade across to it
          fadeTimer = setTimeout(async () => {
            if (bankOn()) {
              setLive(true);
              await new Promise((r) => setTimeout(r, 450)); // the live face wakes up
            }
            const ok = await seedFromFront();
            if (ok) await new Promise((r) => setTimeout(r, 160)); // let the live face render its first matching frames
            release(ok);
            if (bankOn() && !current) backToBank();
          }, 400);
        } else if (bankOn()) {
          backToBank();
        }
        resolve(ok);
      };
      voice.addEventListener("ended", () => {
        tail = setTimeout(() => finish(true), 300);
      });
      voice.addEventListener("error", () => {
        note(`The voice for ${label} could not be loaded.`);
        finish(false);
      });
      whenRest()
        .then(() => startVideo(videoUrl))
        .then(() => voice.play().catch((e) => note(`Sound was blocked by the browser (${e?.name || e}) — click the page once and try again.`)))
        .catch((e) => {
          note(`The ${label} picture could not start (${e?.name || e}).`);
          finish(false);
        });
      setTimeout(() => finish(true), maxMs); // never get stuck
    });
    return current;
  }

  function play(kind) {
    const clip = pick(kind);
    if (!clip) return Promise.resolve(false);
    if (current) return current;
    return run({
      videoUrl: `${base}/cinematic/${clip.video}`,
      voiceUrl: `${base}/cinematic/src/${clip.id}.wav`,
      maxMs: Math.max(6000, (clip.seconds || 3) * 1000 + 4000),
      label: `"${clip.id}" clip`,
    });
  }

  function playReply(videoUrl, blob) {
    if (current) return current.then(() => playReply(videoUrl, blob));
    return run({ videoUrl, voiceUrl: URL.createObjectURL(blob), maxMs: 90000, label: "reply" });
  }

  /** Hide the photo layer (used when the animated face has to speak because no video was available). */
  function release(matched = false) {
    for (const v of layers) {
      v.style.transition = matched ? "opacity .06s linear" : "opacity .3s ease"; // a match is a cut: nothing to blend
      v.style.opacity = "0";
    }
    front = -1;
  }

  return {
    clips,
    play,
    playReply,
    release,
    playIfIdle: (kind) => (current ? Promise.resolve(false) : play(kind)),
    whenIdle: () => current || Promise.resolve(),
    get busy() {
      return Boolean(current);
    },
    settle,
    get hasIdle() {
      return idleOk;
    },
    dispose() {
      disposed = true;
      clearTimeout(matteTimer);
      clearTimeout(bankTimer);
      clearTimeout(resumeTimer);
      clearTimeout(bankLoadTimer);
      bankVideo.remove();
      pickerEl?.remove();
      bg.remove();
      fg.remove();
      clearTimeout(fadeTimer);
      clearInterval(idlePoll);
      stopFrames();
      idleImg?.remove();
      baseStill.remove();
      for (const v of layers) {
        try {
          v.pause();
          v.remove();
        } catch {
          /* ignore */
        }
      }
    },
  };
}
