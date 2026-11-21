/**
 * Photo-real layer drawn over the live avatar: pre-rendered "cinematic" clips (greeting, thinking, reactions,
 * closing) and live-rendered reply videos, all from the same wide photo so the framing never jumps.
 *
 * ml/cinematic_build.py renders the clips into data/profiles/<user>/media/cinematic/ (manifest.json + the spoken line as
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
  // /data/profiles/<person>/media/scan/x.jpg -> base = /data/profiles/<person>/media, slug = <person>
  const base = (sample.match(/^(.*\/profiles\/[^/]+\/media)\//) || [])[1];
  if (!base) return null;
  const slug = base.split("/").slice(-2)[0];

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
  const matteUrl = `${base}/cinematic/talk_matte.png`;
  const sourceUrl = `${base}/cinematic/source.jpg`;
  const BG_KEY = `avatar-bg4-${slug}`;
  let matteReady = false;
  let bgActive = false;
  const BACKGROUNDS = {
    original: { label: "Room" },
    gray: { label: "Gray", css: "#6e7680" },
    studio: { label: "Studio", css: "radial-gradient(ellipse at 50% 30%, #3b4252 0%, #232838 45%, #11141c 100%)" },
    dark: { label: "Dark", css: "#15171c" },
    blue: { label: "Blue", css: "linear-gradient(160deg, #3a5a8c 0%, #1d2f52 100%)" },
    warm: { label: "Warm", css: "linear-gradient(160deg, #c9a27c 0%, #7a5a45 100%)" },
    green: { label: "Green", css: "linear-gradient(160deg, #4f7a62 0%, #26413a 100%)" },
    sunset: { label: "Sunset", css: "linear-gradient(180deg, #f0a35e 0%, #b4527a 55%, #3b2f5c 100%)" },
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
    bgActive = on;
    if (!on) showAlpha(false);
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
  let bgKey = "original"; // the default look: the room, as in the photo
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
  // lookForMatte(); // the background picker needs a cut-out of the person; replies are now used as FlashHead draws them

  // Find out first whether saved waiting animations exist, so the very first picture is already the right one
  // (no photo -> rest frame swap, no live face flashing in just to be hidden again).
  let earlyBank = [];
  try {
    const r = await fetch(`${base}/cinematic/idle_bank/manifest.json`, { cache: "no-store" });
    if (r.ok) earlyBank = (await r.json()).clips || [];
  } catch {
    /* none yet */
  }
  const bankFromStart = earlyBank.length >= 6;
  let bankInCropView = false; // the saved animations are FlashHead's own (already the head-and-shoulders view)
  try {
    bankInCropView = (await (await fetch(`${base}/cinematic/idle_bank/manifest.json`, { cache: "no-store" })).json()).space === "crop";
  } catch {
    /* no set yet */
  }

  // The talking videos are FlashHead's head-and-shoulders crop of the photo. The full-photo pictures (waiting animations) are zoomed
  // to the same crop (talk_meta.json), and the still picture is FlashHead's own rest frame, so every clip starts and ends alike.
  let cropMeta = null;
  let hasRest = false;
  try {
    const r = await fetch(`${base}/cinematic/talk_meta.json`, { cache: "no-store" });
    if (r.ok) cropMeta = await r.json();
    hasRest = Boolean(cropMeta) && (await fetch(`${base}/cinematic/talk_rest.jpg`, { method: "HEAD", cache: "no-store" })).ok;
  } catch {
    /* no talking crop yet: the pictures stay as they were */
  }
  const crop = (el) => {
    if (!cropMeta) return el;
    const [x1, y1, x2] = cropMeta.box;
    const [W, H] = cropMeta.size;
    el.style.transformOrigin = "0 0";
    el.style.transform = `scale(${W / (x2 - x1)}) translate(${(-x1 / W) * 100}%, ${(-y1 / H) * 100}%)`;
    return el;
  };

  // Base layer: the same prepared photo every clip and reply is made from, so size and framing always match.
  const baseStill = new Image();
  baseStill.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;border-radius:inherit;z-index:0;pointer-events:none;";
  baseStill.src = hasRest ? `${base}/cinematic/talk_rest.jpg?v=${Math.round(cropMeta.stamp[0])}` : bankFromStart ? `${base}/cinematic/idle_bank/rest.jpg` : `${base}/cinematic/source.jpg`;
  if (!hasRest) crop(baseStill);
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
  if (!bankInCropView) crop(bankVideo);
  fg.appendChild(bankVideo);
  // The transparent twin of each animation: its cut-out is worked out per frame, so with a replaced background the outline
  // follows the head. It sits above the (statically masked) stack, which is hidden while it plays.
  const alphaVideo = document.createElement("video");
  alphaVideo.playsInline = true;
  alphaVideo.muted = true;
  alphaVideo.preload = "auto";
  alphaVideo.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;border-radius:inherit;opacity:0;pointer-events:none;z-index:2;";
  if (!bankInCropView) crop(alphaVideo);
  figure.appendChild(alphaVideo);
  const showAlpha = (on) => {
    alphaVideo.style.opacity = on ? "1" : "0";
    fg.style.visibility = on ? "hidden" : "visible";
  };
  let bank = earlyBank;
  let bankPlaying = false;
  let mainActive = false; // a greeting / reply is on screen (or about to be)
  let mainIdleSince = Date.now(); // when the last greeting / reply sentence finished
  let lastDiag = 0;
  let thinkUntil = 0; // while a reply is being prepared: only thoughtful animations
  let liveWanted = false; // with saved animations the live face stays hidden from the first moment (showLive is false while they exist)
  let bankTimer = 0;
  let resumeTimer = 0;
  let bankLoadTimer = 0;
  let lastBank = "";
  let bankWasOn = false;
  let restWaiters = [];
  let bankFails = 0;
  let lastBankEnd = Date.now();
  let lastProgress = Date.now();
  const bankOn = () => bank.length >= 6 && bankFails < 3; // after repeated failures the live face does the waiting
  bankVideo.addEventListener("timeupdate", () => (lastProgress = Date.now()));
  // watchdog: if the chain of animations ever stalls or breaks, restart it (or hand over to the live face)
  const bankWatch = setInterval(() => {
    if (disposed) return;
    // a greeting / reply finished a while ago and nothing new started (e.g. waiting for the next sentence): keep animating
    if (mainActive && !current && bankOn() && Date.now() - mainIdleSince > 4500) {
      mainActive = false;
      clearTimeout(resumeTimer);
      setLive(false);
      bankVideo.style.opacity = "0";
      tickBank();
    }
    if (mainActive) return;
    // proof of life: saved loops chain with gaps under a second. If none has played for 8 s, they cannot play here:
    // use the live face (which always moves) instead of freezing on a still photo.
    const lastAlive = Math.max(lastProgress, lastBankEnd, mainIdleSince);
    if (bankWasOn && bankOn() && Date.now() - lastAlive > 8000) {
      bankFails = 3;
      note("The saved waiting animations would not play in this browser; using the live face instead.");
    }
    if (bank.length >= 6 && !bankPlaying && Date.now() - lastBankEnd > 9000 && Date.now() - lastDiag > 15000) {
      lastDiag = Date.now();
      note(`Waiting animations idle (clips ${bank.length}, fails ${bankFails}, think ${Date.now() < thinkUntil})`);
    }
    if (bank.length >= 6 && bankFails >= 3 && !liveWanted) {
      setLive(true);
      return;
    }
    if (!bankOn()) return;
    if (bankPlaying && Date.now() - lastProgress > 5000) {
      bankFails += 1;
      bankPlaying = false;
      resolveRest();
    }
    if (!bankPlaying && Date.now() - lastBankEnd > 3000) tickBank();
  }, 2500);
  const showLive = () => !bankOn() || liveWanted;
  const resolveRest = () => {
    const w = restWaiters;
    restWaiters = [];
    for (const f of w) f();
  };
  const whenRest = (maxMs = 14000) => // wait for the animation on screen to FINISH (the longest are about 7 s); the cap only guards against a stuck one
    bankPlaying ? new Promise((res) => {
      restWaiters.push(res);
      setTimeout(res, maxMs);
    }) : Promise.resolve();

  const recentBank = [];
  const goofyPlays = {};
  let lastGoofy = Date.now(); // none right after the page opens
  function pickBank() {
    const thinking = Date.now() < thinkUntil;
    // never repeat anything from the last several clips (as long as there are enough to choose from)
    const keep = Math.min(8, Math.max(1, Math.floor(bank.length / 3)));
    const needAlpha = bgActive; // with a replaced background only animations that carry their own cut-out can be used
    const goofyOk = Date.now() - lastGoofy > 150000; // a goofy face at most every couple of minutes, and never while thinking
    const candidates = bank.filter(
      (c) =>
        !recentBank.slice(-keep).includes(c.id) &&
        (!thinking || c.think || c.kind === "breathe") &&
        (!c.ephemeral || (goofyOk && !thinking && (goofyPlays[c.id] || 0) < 2)),
    );
    const withAlpha = needAlpha ? candidates.filter((c) => c.webm) : candidates; // prefer clips with their own cut-out
    const pool = withAlpha.length ? withAlpha : candidates;
    if (!pool.length) return null;
    const weight = (c) => (c.kind === "breathe" ? (thinking ? 2 : 6) : c.kind === "yawn" ? 0.4 : c.ephemeral ? 0.6 : c.think && thinking ? 3 : 1.5);
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
    recentBank.push(c.id);
    if (c.ephemeral) {
      goofyPlays[c.id] = (goofyPlays[c.id] || 0) + 1;
      lastGoofy = Date.now();
    }
    if (recentBank.length > 12) recentBank.shift();
    try {
      const useAlpha = bgActive && Boolean(c.webm);
      const vid = useAlpha ? alphaVideo : bankVideo;
      vid.src = `${bankBase}/${useAlpha ? c.webm : c.file}`;
      vid.currentTime = 0;
      await vid.play();
      if (useAlpha) {
        bankVideo.style.opacity = "0";
        showAlpha(true);
      } else {
        showAlpha(false);
        bankVideo.style.opacity = "1";
      }
      await new Promise((res) => {
        vid.onended = res;
        vid.onerror = res;
        setTimeout(res, (c.dur + 1) * 1000);
      });
    } catch (e) {
      // AbortError = the browser cancelled this clip because another video took over (a reply, a clip): not a failure
      if (e?.name !== "AbortError") {
        bankFails += 1;
        note(`Waiting animation problem (${e?.name || e}); ${bankFails}/3`);
      }
    }
    bankPlaying = false;
    lastBankEnd = Date.now();
    resolveRest();
  }
  const pingBusy = (sec = 45) => {
    try {
      fetch(`${IDLE}/busy?sec=${sec}`, { method: "POST", mode: "no-cors" }).catch(() => {});
    } catch {
      /* ignore */
    }
  };
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
      lastBankEnd = lastProgress = Date.now(); // the proof-of-life clock starts now, not when the page opened (first runs wait a while)
      if (!hasRest) baseStill.src = `${bankBase}/rest.jpg`; // the exact frame every animation starts and ends on (FlashHead's own rest frame is already showing)
      setLive(false);
      tickBank();
    }
    if (!disposed) bankLoadTimer = setTimeout(loadBank, bank.length < 160 ? 20000 : 120000); // new animations keep arriving
  }
  /** After a greeting or reply: let the live face carry on from the last frame, then settle back to the pre-made animations. */
  function backToBank() {
    clearTimeout(resumeTimer);
    // ease the live face to the rest pose first (keeping the pose it just inherited fading out), so the cut is invisible
    setTimeout(() => {
      if (liveWanted && !mainActive) return;
      fetch(`${IDLE}/settle?on=1&keep=1`, { method: "POST", mode: "no-cors" }).catch(() => {});
    }, 1600);
    resumeTimer = setTimeout(() => {
      mainActive = false;
      if (bankOn()) {
        setLive(false);
        bankVideo.style.opacity = "0";
        showAlpha(false);
        tickBank();
        fetch(`${IDLE}/settle?on=0`, { method: "POST", mode: "no-cors" }).catch(() => {});
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
  let idleOk = false;
  let idleImg = null;
  let idlePoll = 0;
  let lastFrames = -1;
  let lastClaim = 0;
  let claims = 0;
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
    const next = (delay) => setTimeout(() => alive && (img.src = `${IDLE}/frame.jpg?user=${encodeURIComponent(slug)}&n=${++frameNo}`), delay);
    img.onload = () => {
      lastFrameAt = Date.now();
      next(30);
    };
    img.onerror = () => {
      img.style.opacity = "0"; // not our person (or the service is busy): show nothing rather than someone else
      next(600);
    };
    next(0);
    const poll = async () => {
      try {
        const r = await fetch(`${IDLE}/health`, { cache: "no-store" });
        const j = await r.json();
        if (j.ok && j.user !== slug && document.hasFocus() && Date.now() - lastClaim > 20000 && claims < 3) {
          // the live-face service is holding someone else (a restart, another tab): take it back, but only from the window
          // being used, at most a few times, so two open tabs cannot trade it back and forth
          lastClaim = Date.now();
          claims += 1;
          fetch(`${IDLE}/use?user=${encodeURIComponent(slug)}`, { method: "POST", mode: "no-cors" }).catch(() => {});
        }
        idleOk = false; // the live face is retired: waiting animations come from the saved set, replies from FlashHead
        if (idleOk) {
          if (Date.now() - lastFrameAt > 3000) img.src = `${IDLE}/frame.jpg?user=${encodeURIComponent(slug)}&n=${++frameNo}`; // frames stopped arriving: kick it
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
      if (on) pingBusy(8); // a blip of 'thinking' (the mic hears noise too) only pauses new animations briefly
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
    setTimeout(() => {
      bankVideo.style.opacity = "0";
      showAlpha(false);
    }, 220);
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
  function run({ videoUrl, voiceUrl, maxMs, label, onStart }) {
    mainActive = true;
    pingBusy();
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
        mainIdleSince = Date.now();
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
          // no live face to hand over to: hold the last frame briefly, then clear it so the waiting animations show
          clearTimeout(fadeTimer);
          fadeTimer = setTimeout(() => {
            if (current) return;
            release(false);
            clearTimeout(resumeTimer);
            mainActive = false;
            bankVideo.style.opacity = "0";
            tickBank();
          }, 500);
        }
        resolve(ok);
      };
      voice.addEventListener("ended", () => {
        // The reply video runs a little longer than the speech so the head can settle back onto the rest frame. Let it finish:
        // freezing it now would hold the head mid-way and then jump to the rest picture.
        const v = frontVideo();
        if (v && !v.ended && !v.paused && Number.isFinite(v.duration)) {
          const left = Math.max(0, v.duration - v.currentTime) * 1000;
          v.addEventListener(
            "ended",
            () => {
              clearTimeout(tail);
              tail = setTimeout(() => finish(true), 120);
            },
            { once: true },
          );
          tail = setTimeout(() => finish(true), left + 800); // never wait forever
        } else {
          tail = setTimeout(() => finish(true), 300);
        }
      });
      voice.addEventListener("error", () => {
        note(`The voice for ${label} could not be loaded.`);
        finish(false);
      });
      whenRest()
        .then(() => startVideo(videoUrl))
        .then(() =>
          voice
            .play()
            .catch((e) => note(`Sound was blocked by the browser (${e?.name || e}) — click the page once and try again.`))
            .finally(() => onStart?.(voice.duration)),
        )
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

  function playReply(videoUrl, blob, onStart) {
    if (current) return current.then(() => playReply(videoUrl, blob, onStart));
    return run({ videoUrl, voiceUrl: URL.createObjectURL(blob), maxMs: 90000, label: "reply", onStart });
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
      clearInterval(bankWatch);
      bankVideo.remove();
      alphaVideo.remove();
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
