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

  // Base layer: the same prepared photo every clip and reply is made from, so size and framing always match.
  const baseStill = new Image();
  baseStill.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;border-radius:inherit;z-index:0;pointer-events:none;";
  baseStill.src = `${base}/cinematic/source.jpg`;
  figure.appendChild(baseStill);
  const rawStill = stageEl.querySelector("#avatar-still");
  if (rawStill) rawStill.style.display = "none";

  const makeVideo = () => {
    const v = document.createElement("video");
    v.playsInline = true;
    v.muted = true; // picture only; voices come from audio elements
    v.preload = "auto";
    v.style.cssText =
      "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;border-radius:inherit;opacity:0;pointer-events:none;transition:opacity .2s ease;";
    figure.appendChild(v);
    return v;
  };
  const layers = [makeVideo(), makeVideo()];
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
    figure.appendChild(img);
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
      img.style.opacity = idleOk ? "1" : "0";
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
            await seedFromFront();
            release();
          }, 500);
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
      startVideo(videoUrl)
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
  function release() {
    for (const v of layers) {
      v.style.transition = "opacity .3s ease";
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
