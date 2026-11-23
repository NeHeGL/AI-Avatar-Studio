/**
 * The avatar's picture: the person's prepared photo, the waiting animations that play while nobody is talking, and the lip-sync
 * reply videos rendered by the local FlashHead service. Everything is made from the same prepared photo, so the framing never jumps.
 *
 * Files (data/profiles/<person>/cache/face/):
 *   talk_rest.jpg, talk_meta.json  FlashHead's rest frame of the photo, and the crop it uses
 *   idle_bank/manifest.json + calm_NNN.mp4  the waiting animations (every one starts and ends on the same rest frame)
 *
 * Two <video> layers cross-fade for replies, so the picture never drops out between one sentence and the next. Pictures play MUTED
 * (browsers always allow that); the voice plays through a normal audio element.
 *
 *   const avatarFace = await loadAvatarFace(stageEl, profile, { onNote });   // null when the person has no prepared photo
 *   await avatarFace.whenIdle();  await avatarFace.playReply(videoUrl, wavBlob);  avatarFace.release();  avatarFace.settle(true);  avatarFace.dispose();
 */
export async function loadAvatarFace(stageEl, profile, opts = {}) {
  const note = (m) => {
    try {
      console.warn("[avatar-face]", m);
      opts.onNote?.(m);
    } catch {
      /* ignore */
    }
  };
  const frames = (profile.scan_frames?.length ? profile.scan_frames : profile.scan_previews) || [];
  const sample = profile.avatar_face_url || frames[0]?.url || "";
  // /data/profiles/<person>/photo/x.jpg -> base = /data/profiles/<person>/cache
  const person = (sample.match(/^(.*\/profiles\/[^/]+)\/photo\//) || [])[1];
  if (!person) return null;
  const base = person + "/cache";

  // The server prepares the person's photo (from whatever photos exist).
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

  const figure = stageEl.querySelector("#avatar");
  if (!figure) return null;
  figure.style.position = "relative";
  // Everything that moves lives in `fg`.
  const fg = document.createElement("div");
  fg.style.cssText = "position:absolute;inset:0;border-radius:inherit;z-index:1;overflow:hidden;";
  figure.appendChild(fg);
  let disposed = false;

  // Find out first whether saved waiting animations exist, so the very first picture is already the right one.
  let earlyBank = [];
  let bankVer = ""; // build time of the saved animations: part of every clip address, so an older set is never replayed from the browser's cache
  let bankInCropView = false; // the saved animations are FlashHead's own (already the head-and-shoulders view)
  try {
    const r = await fetch(`${base}/face/idle_bank/manifest.json`, { cache: "no-store" });
    if (r.ok) {
      const j = await r.json();
      earlyBank = j.clips || [];
      bankVer = String(j.built || "");
      bankInCropView = j.space === "crop";
    }
  } catch {
    /* none yet */
  }
  const bankFromStart = earlyBank.length >= 6;

  // The talking videos are FlashHead's head-and-shoulders crop of the photo. The still picture is FlashHead's own rest frame, so every
  // clip starts and ends alike.
  let cropMeta = null;
  let hasRest = false;
  try {
    const r = await fetch(`${base}/face/talk_meta.json`, { cache: "no-store" });
    if (r.ok) cropMeta = await r.json();
    hasRest = Boolean(cropMeta) && (await fetch(`${base}/face/talk_rest.jpg`, { method: "HEAD", cache: "no-store" })).ok;
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
  baseStill.src = hasRest ? `${base}/face/talk_rest.jpg?v=${Math.round(cropMeta.stamp[0])}` : bankFromStart ? `${base}/face/idle_bank/rest.jpg?v=${bankVer}` : `${base}/face/source.jpg`;
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
  const bankBase = `${base}/face/idle_bank`;
  const bankVideo = document.createElement("video");
  bankVideo.playsInline = true;
  bankVideo.muted = true;
  bankVideo.preload = "auto";
  bankVideo.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;border-radius:inherit;opacity:0;pointer-events:none;z-index:1;";
  if (!bankInCropView) crop(bankVideo);
  fg.appendChild(bankVideo);
  let bank = earlyBank;
  let bankPlaying = false;
  let mainActive = false; // a reply is on screen (or about to be)
  let mainIdleSince = Date.now(); // when the last reply sentence finished
  let thinkUntil = 0; // while a reply is being prepared, the next animation follows at once
  let bankTimer = 0;
  let resumeTimer = 0;
  let bankLoadTimer = 0;
  let bankWasOn = false;
  let restWaiters = [];
  let bankFails = 0;
  let lastBankEnd = Date.now();
  let lastProgress = Date.now();
  const bankOn = () => bank.length >= 6 && bankFails < 3; // after repeated failures the still picture stays up
  bankVideo.addEventListener("timeupdate", () => (lastProgress = Date.now()));
  // watchdog: if the chain of animations ever stalls or breaks, restart it
  const bankWatch = setInterval(() => {
    if (disposed) return;
    // a reply finished a while ago and nothing new started (e.g. waiting for the next sentence): keep animating
    if (mainActive && !current && bankOn() && Date.now() - mainIdleSince > 4500) {
      mainActive = false;
      clearTimeout(resumeTimer);
      bankVideo.style.opacity = "0";
      tickBank();
    }
    if (mainActive) return;
    // proof of life: saved loops chain with gaps under a second. If none has played for 8 s, they cannot play here.
    const lastAlive = Math.max(lastProgress, lastBankEnd, mainIdleSince);
    if (bankWasOn && bankOn() && Date.now() - lastAlive > 8000) {
      bankFails = 3;
      note("The saved waiting animations would not play in this browser; showing the still picture instead.");
    }
    if (!bankOn()) return;
    if (bankPlaying && Date.now() - lastProgress > 5000) {
      bankFails += 1;
      bankPlaying = false;
      resolveRest();
    }
    if (!bankPlaying && Date.now() - lastBankEnd > 3000) tickBank();
  }, 2500);
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
  function pickBank() {
    // never repeat anything from the last several clips (as long as there are enough to choose from)
    const keep = Math.min(8, Math.max(1, Math.floor(bank.length / 3)));
    const pool = bank.filter((c) => !recentBank.slice(-keep).includes(c.id));
    return pool.length ? pool[Math.floor(Math.random() * pool.length)] : null;
  }
  async function playBank(c) {
    bankPlaying = true;
    recentBank.push(c.id);
    if (recentBank.length > 12) recentBank.shift();
    try {
      bankVideo.src = `${bankBase}/${c.file}?v=${bankVer}`;
      bankVideo.currentTime = 0;
      await bankVideo.play();
      bankVideo.style.opacity = "1";
      await new Promise((res) => {
        bankVideo.onended = res;
        bankVideo.onerror = res;
        setTimeout(res, (c.dur + 1) * 1000);
      });
    } catch (e) {
      // AbortError = the browser cancelled this clip because another video took over (a reply): not a failure
      if (e?.name !== "AbortError") {
        bankFails += 1;
        note(`Waiting animation problem (${e?.name || e}); ${bankFails}/3`);
      }
    }
    bankPlaying = false;
    lastBankEnd = Date.now();
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
  async function loadBank() {
    try {
      const r = await fetch(`${bankBase}/manifest.json`, { cache: "no-store" });
      if (r.ok) {
        const j = await r.json();
        bank = j.clips || [];
        bankVer = String(j.built || "");
        if (j.space === "crop" && !bankInCropView) {
          // the animations arrived after the page opened: they are already the head-and-shoulders view, so the zoom set up for a full photo
          // must go, or the face is cropped twice (zoomed in, blurred and washed out)
          bankInCropView = true;
          for (const el of [bankVideo, ...(hasRest ? [] : [baseStill])]) {
            el.style.transform = "";
            el.style.transformOrigin = "";
          }
        }
      }
    } catch {
      /* no bank yet: the still picture stays */
    }
    if (bankOn() && !bankWasOn) {
      bankWasOn = true;
      lastBankEnd = lastProgress = Date.now(); // the proof-of-life clock starts now, not when the page opened (first runs wait a while)
      if (!hasRest) baseStill.src = `${bankBase}/rest.jpg?v=${bankVer}`; // the exact frame every animation starts and ends on (FlashHead's own rest frame is already showing)
      tickBank();
    }
    if (!disposed) bankLoadTimer = setTimeout(loadBank, bank.length < 160 ? 20000 : 120000); // new animations keep arriving
  }
  void loadBank();
  let front = -1; // index of the layer currently showing, -1 = none yet
  let fadeTimer = 0;

  /** on=true: a reply is being prepared, so the next waiting animation follows at once and the reply video can take over from a calm frame. */
  function settle(on) {
    if (!bankOn()) return;
    thinkUntil = on ? Date.now() + 20000 : 0;
    if (on) tickBank();
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

  let current = null; // promise of the running reply

  /** Common run-loop: show a video, play a voice, resolve shortly after the voice ends. The last frame stays up. */
  function run({ videoUrl, voiceUrl, maxMs, label, onStart }) {
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
          frontVideo()?.pause(); // hold the last frame (the picture of you)
        } catch {
          /* ignore */
        }
        if (voiceUrl.startsWith("blob:")) URL.revokeObjectURL(voiceUrl);
        current = null;
        mainIdleSince = Date.now();
        if (bankOn()) {
          // hold the last frame briefly, then clear it so the waiting animations show
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

  function playReply(videoUrl, blob, onStart) {
    if (current) return current.then(() => playReply(videoUrl, blob, onStart));
    return run({ videoUrl, voiceUrl: URL.createObjectURL(blob), maxMs: 90000, label: "reply", onStart });
  }

  /** Hide the reply layers (used when no video was available for a sentence: the voice plays over the still picture). */
  function release(matched = false) {
    for (const v of layers) {
      v.style.transition = matched ? "opacity .06s linear" : "opacity .3s ease"; // a match is a cut: nothing to blend
      v.style.opacity = "0";
    }
    front = -1;
  }

  return {
    playReply,
    release,
    whenIdle: () => current || Promise.resolve(),
    settle,
    dispose() {
      disposed = true;
      clearTimeout(bankTimer);
      clearTimeout(resumeTimer);
      clearTimeout(bankLoadTimer);
      clearTimeout(fadeTimer);
      clearInterval(bankWatch);
      bankVideo.remove();
      fg.remove();
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
